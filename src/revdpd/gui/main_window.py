"""Main window of the revdpd back-mapping GUI."""
from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QSettings, Qt, QThread, QUrl, Signal, Slot
from PySide6.QtGui import QAction, QColor, QDesktopServices, QIcon, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDockWidget, QDoubleSpinBox,
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea,
    QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout, QWidget,
)

from .. import __version__
from ..core.backmap import BackmapSettings, Fitter, estimate_scale, place_cluster
from ..core.cg_system import CGSystem
from ..core.lammps_runner import find_lammps
from ..core.mapping import BeadMapping, atom_owners, auto_linear_mapping, reverse_mapping
from ..core.pipeline import (
    BUILTIN_SPC, Job, MinimizeSettings, OverlapSettings, Project, SpeciesAssignment, load_template,
    resolve_species, run_backmapping,
)
from ..io.lammps_data import read_lammps_data
from ..io.lammps_writer import OutputSettings
from ..io.moltemplate import AAMolecule, parse_forcefield, parse_molecule, spc_water
from .mol_view import MoleculeView

BEAD_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6",
               "#bfef45", "#469990", "#9a6324", "#800000", "#808000", "#000075", "#fabed4",
               "#ffd8b1", "#aaffc3", "#dcbeff", "#a9a9a9"]
ELEMENT_COLORS = {"H": "#eeeeee", "C": "#606060", "N": "#3050f8", "O": "#ff0d0d", "P": "#ff8000",
                  "S": "#e6c200", "F": "#90e050", "Cl": "#1ff01f", "Br": "#a62929", "Na": "#ab5cf2"}

HELP = """<h3>Workflow</h3>
<ol>
<li><b>Load the CG system</b> (LAMMPS data, DPD units). Choose the atom style
(<i>angle</i> or <i>full</i>; <i>auto</i> reads the <code>Atoms # style</code> comment).
Molecules are split by molecule ID and identical molecules are grouped into species.</li>
<li><b>Select a species</b> and load its <b>all-atom moltemplate file</b> (.lt). The force-field
file it inherits from is found automatically in the same folder (or choose it).</li>
<li><b>Map atoms to beads</b>: click a bead in the CG panel (or press 1-9), then click heavy atoms
in the all-atom panel. Clicking again un-assigns. <b>Shift+drag</b> box-selects.
Hydrogens follow the heavy atom they are bonded to.
<i>Auto (chain)</i> proposes a mapping for chain-like molecules.</li>
<li><b>Scale</b>: Angstrom per DPD length unit. <i>Estimate</i> compares bead distances of the
template with CG bond lengths. Tick <i>Overlay fit</i> to preview the fitted molecule.</li>
<li><b>Run</b>: every mapped species is fitted onto all its CG molecules (Kabsch fit of the bead
centres - the orientation of every molecule is kept), optional rigid-body overlap removal,
LAMMPS files are written together with a restrained, gradual relaxation script
(bonded-only minimisation, soft push-off, minimisations with decreasing position restraints,
restrained MD with increasing time step, final unrestrained minimisation), which can be run
directly.</li>
<li><b>Solvent</b>: a single-bead species (e.g. DPD water) can be replaced by N_m molecules per
bead (<i>Molecules per bead</i>); <i>Built-in SPC water</i> provides a water template.
Alternatively untick the solvent species and solvate the all-atom system afterwards.</li>
</ol>
<h3>Mouse</h3>
Left drag: rotate &nbsp; Ctrl+left drag: roll &nbsp; Right/middle drag: pan &nbsp;
Wheel: zoom &nbsp; Double-click: reset view"""


@dataclass
class SpeciesState:
    aa: AAMolecule | None = None
    mapping: BeadMapping | None = None
    aa_path: str = ""
    ff_path: str = ""
    enabled: bool = True
    copies: int = 1


class Worker(QObject):
    log = Signal(str)
    progress = Signal(str, int, int)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.stop_requested = False

    @Slot()
    def run(self):
        try:
            res = self.fn(self.log.emit, lambda name, i, n: self.progress.emit(name, i, n),
                          lambda: self.stop_requested)
            self.done.emit(res)
        except Exception as exc:  # noqa: BLE001 - report everything to the GUI
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


def _swatch(color: str) -> QIcon:
    pm = QPixmap(14, 14)
    pm.fill(QColor(color))
    return QIcon(pm)


def _hline(*widgets) -> QWidget:
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    for x in widgets:
        if isinstance(x, int):
            lay.addStretch(x)
        else:
            lay.addWidget(x)
    return w


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"revdpd {__version__} - DPD to all-atom back-mapping")
        self.resize(1500, 900)
        self.settings = QSettings("revdpd", "revdpd")
        self.cg: CGSystem | None = None
        self.states: dict[int, SpeciesState] = {}
        self.cur: int | None = None
        self.active_bead = 0
        self.thread: QThread | None = None
        self.worker: Worker | None = None
        self._bond_len_cache: dict[int, float] = {}
        self._build_ui()
        self._build_menu()
        self._set_species_widgets_enabled(False)

    # ================================================================== UI
    def _build_ui(self):
        # ---------- central: two viewers
        self.aa_view = MoleculeView()
        self.aa_view.placeholder = "Load an all-atom template (.lt) for the selected species"
        self.cg_view = MoleculeView()
        self.cg_view.placeholder = "Load a CG LAMMPS data file"
        self.chk_h = QCheckBox("Hydrogens")
        self.chk_h.setChecked(True)
        self.chk_aa_labels = QCheckBox("Labels")
        self.cmb_color = QComboBox()
        self.cmb_color.addItems(["Color: mapping", "Color: element"])
        self.chk_cg_labels = QCheckBox("Labels")
        self.chk_cg_labels.setChecked(True)
        self.chk_overlay = QCheckBox("Overlay fit")
        self.chk_overlay.setToolTip("Show the all-atom template fitted onto this CG molecule")
        self.chk_overlay.setChecked(True)
        self.spn_instance = QSpinBox()
        self.spn_instance.setPrefix("molecule ")
        self.spn_instance.setMinimum(1)
        self.btn_rand_inst = QToolButton()
        self.btn_rand_inst.setText("Random")
        self.lbl_aa_title = QLabel("<b>All-atom</b>")
        self.lbl_cg_title = QLabel("<b>Coarse-grained (DPD)</b>")

        aa_panel = QWidget()
        la = QVBoxLayout(aa_panel)
        la.addWidget(_hline(self.lbl_aa_title, 1, self.chk_h, self.chk_aa_labels, self.cmb_color))
        la.addWidget(self.aa_view, 1)
        cg_panel = QWidget()
        lc = QVBoxLayout(cg_panel)
        lc.addWidget(_hline(self.lbl_cg_title, 1, self.chk_cg_labels, self.chk_overlay,
                            self.spn_instance, self.btn_rand_inst))
        lc.addWidget(self.cg_view, 1)
        split = QSplitter(Qt.Horizontal)
        split.addWidget(aa_panel)
        split.addWidget(cg_panel)
        split.setSizes([1000, 1000])
        self.setCentralWidget(split)

        # ---------- left dock: setup + mapping
        left = QWidget()
        L = QVBoxLayout(left)

        g1 = QGroupBox("1  CG system (LAMMPS data)")
        f1 = QFormLayout(g1)
        self.ed_cg = QLineEdit()
        b = QToolButton()
        b.setText("...")
        b.clicked.connect(self.browse_cg)
        f1.addRow("File", _hline(self.ed_cg, b))
        self.cmb_style = QComboBox()
        self.cmb_style.addItems(["auto", "angle", "full", "molecular", "bond"])
        self.cmb_style.setToolTip("LAMMPS atom style of the data file (column layout of the Atoms section)")
        self.cmb_split = QComboBox()
        self.cmb_split.addItems(["auto", "molecule ID", "bonds"])
        self.btn_load_cg = QPushButton("Load")
        self.btn_load_cg.clicked.connect(self.load_cg)
        f1.addRow("Atom style", _hline(self.cmb_style, QLabel("Molecules by"), self.cmb_split))
        f1.addRow(self.btn_load_cg)
        self.tbl_species = QTableWidget(0, 4)
        self.tbl_species.setHorizontalHeaderLabels(["Species", "Beads", "Count", "All-atom"])
        self.tbl_species.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_species.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tbl_species.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl_species.verticalHeader().setVisible(False)
        self.tbl_species.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tbl_species.setMinimumHeight(130)
        self.tbl_species.itemSelectionChanged.connect(self.on_species_selected)
        self.tbl_species.itemChanged.connect(self.on_species_item_changed)
        f1.addRow(self.tbl_species)
        L.addWidget(g1)

        g2 = QGroupBox("2  All-atom template (moltemplate .lt)")
        f2 = QFormLayout(g2)
        self.ed_aa = QLineEdit()
        b = QToolButton()
        b.setText("...")
        b.clicked.connect(self.browse_aa)
        f2.addRow("Molecule", _hline(self.ed_aa, b))
        self.ed_ff = QLineEdit()
        self.ed_ff.setPlaceholderText("auto-detect (same folder / import)")
        b = QToolButton()
        b.setText("...")
        b.clicked.connect(self.browse_ff)
        f2.addRow("Force field", _hline(self.ed_ff, b))
        self.btn_load_aa = QPushButton("Load for selected species")
        self.btn_load_aa.clicked.connect(self.load_aa)
        self.btn_water = QPushButton("Built-in SPC water")
        self.btn_water.setToolTip("Use an SPC water molecule with the OW/H atom types of the force field\n"
                                  "(GROMOS/ATB). For single-bead solvent species.")
        self.btn_water.clicked.connect(self.load_builtin_water)
        f2.addRow(_hline(self.btn_load_aa, self.btn_water))
        self.spn_copies = QSpinBox()
        self.spn_copies.setRange(1, 100)
        self.spn_copies.setToolTip("Number of all-atom molecules represented by one CG bead (N_m).\n"
                                   "Only for single-bead species, e.g. 3 for a DPD water bead with N_m = 3.")
        self.spn_copies.valueChanged.connect(self.on_copies_changed)
        f2.addRow("Molecules per bead (N_m)", self.spn_copies)
        self.lbl_aa_info = QLabel("")
        self.lbl_aa_info.setWordWrap(True)
        f2.addRow(self.lbl_aa_info)
        L.addWidget(g2)

        g3 = QGroupBox("3  Mapping (heavy atoms -> beads)")
        v3 = QVBoxLayout(g3)
        hint = QLabel("Select a bead (click it in the CG panel, a row below, or press 1-9), then "
                      "click heavy atoms in the all-atom panel. Shift+drag box-selects.")
        hint.setWordWrap(True)
        v3.addWidget(hint)
        self.tbl_beads = QTableWidget(0, 3)
        self.tbl_beads.setHorizontalHeaderLabels(["Bead", "Type", "Atoms"])
        self.tbl_beads.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_beads.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tbl_beads.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl_beads.verticalHeader().setVisible(False)
        self.tbl_beads.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.tbl_beads.setMinimumHeight(170)
        self.tbl_beads.itemSelectionChanged.connect(self.on_bead_row_selected)
        v3.addWidget(self.tbl_beads)
        self.btn_auto = QPushButton("Auto (chain)")
        self.btn_auto.setToolTip("Split the longest heavy-atom chain into consecutive groups")
        self.btn_reverse = QPushButton("Reverse")
        self.btn_clear_bead = QPushButton("Clear bead")
        self.btn_clear_all = QPushButton("Clear all")
        self.btn_auto.clicked.connect(self.auto_map)
        self.btn_reverse.clicked.connect(self.reverse_map)
        self.btn_clear_bead.clicked.connect(lambda: self._edit_mapping(lambda m: m.clear(self.active_bead)))
        self.btn_clear_all.clicked.connect(lambda: self._edit_mapping(lambda m: m.clear()))
        v3.addWidget(_hline(self.btn_auto, self.btn_reverse, self.btn_clear_bead, self.btn_clear_all))
        self.cmb_center = QComboBox()
        self.cmb_center.addItems(["center of mass (incl. H)", "heavy-atom center of mass", "heavy-atom centroid"])
        self.cmb_center.currentIndexChanged.connect(self.on_center_changed)
        v3.addWidget(_hline(QLabel("Bead centre"), self.cmb_center, 1))
        self.lbl_valid = QLabel("")
        self.lbl_valid.setWordWrap(True)
        v3.addWidget(self.lbl_valid)
        L.addWidget(g3)
        L.addStretch(1)

        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setWidget(left)
        dl = QDockWidget("Setup", self)
        dl.setObjectName("setup")
        dl.setWidget(sa)
        dl.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.LeftDockWidgetArea, dl)
        dl.setMinimumWidth(400)

        # ---------- right dock: back-mapping
        right = QWidget()
        R = QVBoxLayout(right)
        g4 = QGroupBox("Placement")
        f4 = QFormLayout(g4)
        self.spn_scale = QDoubleSpinBox()
        self.spn_scale.setRange(0.01, 1000)
        self.spn_scale.setDecimals(3)
        self.spn_scale.setValue(10.0)
        self.spn_scale.setSuffix(" A / unit")
        self.spn_scale.setToolTip("Length of one DPD length unit (r_c) in Angstrom")
        self.btn_est = QPushButton("Estimate")
        self.btn_est.clicked.connect(self.estimate_scale)
        f4.addRow("Scale", _hline(self.spn_scale, self.btn_est))
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["rigid", "rigid + per-bead shift", "per-bead fragments"])
        self.cmb_mode.setToolTip(
            "rigid: one fit of the whole template; its geometry is kept exactly.\n"
            "per-bead shift: additionally translate each atom group onto its bead.\n"
            "per-bead fragments: rotate each bead's atom group towards its neighbour beads and\n"
            "centre it on the bead (follows bent CG molecules; relaxation repairs the joints).")
        f4.addRow("Fit", self.cmb_mode)
        self.spn_flex = QDoubleSpinBox()
        self.spn_flex.setRange(0, 1)
        self.spn_flex.setSingleStep(0.1)
        self.spn_flex.setValue(1.0)
        f4.addRow("Shift weight", self.spn_flex)
        self.chk_spin = QCheckBox("Random spin about axis of linear molecules")
        self.chk_spin.setChecked(True)
        f4.addRow(self.chk_spin)
        self.spn_seed = QSpinBox()
        self.spn_seed.setRange(0, 2**31 - 1)
        self.spn_seed.setValue(2024)
        f4.addRow("Random seed", self.spn_seed)
        R.addWidget(g4)

        self.g_ov = QGroupBox("Overlap removal (rigid bodies)")
        self.g_ov.setCheckable(True)
        self.g_ov.setChecked(True)
        f5 = QFormLayout(self.g_ov)
        self.spn_dmin = QDoubleSpinBox()
        self.spn_dmin.setRange(0.5, 6)
        self.spn_dmin.setSingleStep(0.1)
        self.spn_dmin.setValue(2.5)
        self.spn_dmin.setSuffix(" A")
        f5.addRow("Min. distance", self.spn_dmin)
        self.spn_oviter = QSpinBox()
        self.spn_oviter.setRange(0, 10000)
        self.spn_oviter.setValue(60)
        f5.addRow("Iterations", self.spn_oviter)
        self.chk_heavy = QCheckBox("Heavy atoms only")
        self.chk_heavy.setChecked(True)
        f5.addRow(self.chk_heavy)
        R.addWidget(self.g_ov)

        g6 = QGroupBox("Output")
        f6 = QFormLayout(g6)
        self.ed_out = QLineEdit(str(Path.cwd() / "backmapped"))
        b = QToolButton()
        b.setText("...")
        b.clicked.connect(self.browse_out)
        f6.addRow("Folder", _hline(self.ed_out, b))
        self.ed_base = QLineEdit("system")
        f6.addRow("Base name", self.ed_base)
        self.spn_cut = QDoubleSpinBox()
        self.spn_cut.setRange(4, 30)
        self.spn_cut.setValue(14.0)
        self.spn_cut.setSuffix(" A")
        f6.addRow("Pair cutoff", self.spn_cut)
        self.chk_long = QCheckBox("Long-range electrostatics (kspace)")
        self.chk_long.setChecked(True)
        f6.addRow(self.chk_long)
        R.addWidget(g6)

        self.g_min = QGroupBox("Run relaxation with LAMMPS")
        self.g_min.setCheckable(True)
        self.g_min.setChecked(False)
        f7 = QFormLayout(self.g_min)
        self.ed_lmp = QLineEdit(self.settings.value("lammps_exe", "") or (find_lammps() or ""))
        b = QToolButton()
        b.setText("...")
        b.clicked.connect(self.browse_lmp)
        f7.addRow("Executable", _hline(self.ed_lmp, b))
        self.spn_mpi = QSpinBox()
        self.spn_mpi.setRange(1, 1024)
        f7.addRow("MPI ranks", self.spn_mpi)
        R.addWidget(self.g_min)

        g8 = QGroupBox("Relaxation protocol (written to *.min.in)")
        f8 = QFormLayout(g8)
        self.chk_bonded = QCheckBox("1. Bonded-only minimisation")
        self.chk_bonded.setChecked(True)
        f8.addRow(self.chk_bonded)
        self.chk_soft = QCheckBox("2. Soft-potential push-off")
        self.chk_soft.setChecked(True)
        f8.addRow(self.chk_soft)
        self.chk_restr = QCheckBox("Restrain heavy atoms to back-mapped positions")
        self.chk_restr.setChecked(True)
        f8.addRow(self.chk_restr)
        self.ed_k = QLineEdit("1000 100 10")
        self.ed_k.setToolTip("3. One full-force-field minimisation per restraint constant (kcal/mol/A^2)")
        f8.addRow("3. Restraint k", self.ed_k)
        self.spn_md = QSpinBox()
        self.spn_md.setRange(0, 10**7)
        self.spn_md.setValue(1000)
        self.spn_md.setToolTip("4. Restrained MD steps per time step value (0 = no MD)")
        f8.addRow("4. MD steps / stage", self.spn_md)
        self.ed_dt = QLineEdit("0.2 0.5 1.0")
        self.ed_dt.setToolTip("Time steps (fs) of the successive restrained MD stages")
        f8.addRow("    Time steps (fs)", self.ed_dt)
        self.spn_temp = QDoubleSpinBox()
        self.spn_temp.setRange(1, 2000)
        self.spn_temp.setValue(300)
        self.spn_temp.setSuffix(" K")
        f8.addRow("    Temperature", self.spn_temp)
        self.chk_release = QCheckBox("5. Final minimisation without restraints")
        self.chk_release.setChecked(True)
        f8.addRow(self.chk_release)
        self.spn_steps = QSpinBox()
        self.spn_steps.setRange(10, 10**7)
        self.spn_steps.setValue(5000)
        f8.addRow("Max. min. iterations", self.spn_steps)
        R.addWidget(g8)

        self.btn_run = QPushButton("Back-map system")
        self.btn_run.setMinimumHeight(36)
        f = self.btn_run.font()
        f.setBold(True)
        self.btn_run.setFont(f)
        self.btn_run.clicked.connect(self.run)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop)
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setValue(0)
        self.btn_open_out = QPushButton("Open output folder")
        self.btn_open_out.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self.ed_out.text())))
        R.addStretch(1)
        sr = QScrollArea()
        sr.setWidgetResizable(True)
        sr.setWidget(right)
        holder = QWidget()
        hv = QVBoxLayout(holder)
        hv.setContentsMargins(0, 0, 0, 0)
        hv.addWidget(sr, 1)
        run_box = QWidget()
        rv = QVBoxLayout(run_box)
        rv.addWidget(_hline(self.btn_run, self.btn_stop))
        rv.addWidget(self.progress)
        rv.addWidget(self.btn_open_out)
        hv.addWidget(run_box)
        dr = QDockWidget("Back-mapping", self)
        dr.setObjectName("backmap")
        dr.setWidget(holder)
        dr.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.RightDockWidgetArea, dr)
        dr.setMinimumWidth(330)

        # ---------- bottom dock: log
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(20000)
        db = QDockWidget("Log", self)
        db.setObjectName("log")
        db.setWidget(self.log_box)
        self.addDockWidget(Qt.BottomDockWidgetArea, db)
        self.resizeDocks([db], [130], Qt.Vertical)

        # ---------- signals
        self.aa_view.atomClicked.connect(self.on_aa_clicked)
        self.aa_view.atomsBoxSelected.connect(self.on_aa_box)
        self.aa_view.atomHovered.connect(self.on_aa_hover)
        self.cg_view.atomClicked.connect(self.set_active_bead)
        self.cg_view.atomHovered.connect(self.on_cg_hover)
        self.chk_h.toggled.connect(lambda: self.refresh_aa(reset=False))
        self.chk_aa_labels.toggled.connect(self.aa_view.set_show_labels)
        self.cmb_color.currentIndexChanged.connect(lambda: self.refresh_aa(reset=False))
        self.chk_cg_labels.toggled.connect(self.cg_view.set_show_labels)
        self.cg_view.set_show_labels(True)
        self.chk_overlay.toggled.connect(lambda: self.refresh_cg(reset=False))
        self.spn_instance.valueChanged.connect(lambda: self.refresh_cg(reset=True))
        self.btn_rand_inst.clicked.connect(self.random_instance)
        for w in (self.spn_scale, self.spn_flex):
            w.valueChanged.connect(lambda: self.refresh_cg(reset=False))
        self.cmb_mode.currentIndexChanged.connect(lambda: self.refresh_cg(reset=False))
        for k in range(9):
            sc = QShortcut(QKeySequence(str(k + 1)), self)
            sc.activated.connect(lambda k=k: self.set_active_bead(k))
        self.statusBar().showMessage("Load a CG LAMMPS data file to start")

    def _build_menu(self):
        m = self.menuBar().addMenu("&File")
        for text, slot, key in [
            ("Open CG data...", self.browse_cg, "Ctrl+O"),
            ("Open all-atom template...", self.browse_aa, "Ctrl+Shift+O"),
            (None, None, None),
            ("Load project...", self.load_project, "Ctrl+L"),
            ("Save project...", self.save_project, "Ctrl+S"),
            (None, None, None),
            ("Quit", self.close, "Ctrl+Q"),
        ]:
            if text is None:
                m.addSeparator()
                continue
            a = QAction(text, self)
            a.setShortcut(QKeySequence(key))
            a.triggered.connect(slot)
            m.addAction(a)
        h = self.menuBar().addMenu("&Help")
        a = QAction("Quick guide", self)
        a.setShortcut(QKeySequence("F1"))
        a.triggered.connect(lambda: QMessageBox.information(self, "revdpd - quick guide", HELP))
        h.addAction(a)
        a = QAction("About", self)
        a.triggered.connect(lambda: QMessageBox.about(
            self, "About revdpd",
            f"<b>revdpd {__version__}</b><br>Interactive back-mapping of DPD coarse-grained "
            "LAMMPS systems to all-atom resolution."))
        h.addAction(a)

    # ============================================================= helpers
    def log(self, msg: str):
        self.log_box.appendPlainText(msg)

    def error(self, title: str, msg: str):
        self.log(f"ERROR: {title}: {msg.splitlines()[0] if msg else ''}")
        QMessageBox.critical(self, title, msg)

    def _dir(self, key="last_dir") -> str:
        return self.settings.value(key, str(Path.home()))

    def _remember(self, path: str, key="last_dir"):
        self.settings.setValue(key, str(Path(path).parent))

    @property
    def state(self) -> SpeciesState | None:
        return None if self.cur is None else self.states.setdefault(self.cur, SpeciesState())

    @property
    def species(self):
        return None if (self.cg is None or self.cur is None) else self.cg.species[self.cur]

    def _set_species_widgets_enabled(self, on: bool):
        for w in (self.ed_aa, self.ed_ff, self.btn_load_aa, self.btn_water):
            w.setEnabled(on)
        self.spn_copies.setEnabled(on and self.species is not None and self.species.n_beads == 1)
        self._set_mapping_widgets_enabled(on and self.state is not None and self.state.aa is not None)

    def _set_mapping_widgets_enabled(self, on: bool):
        for w in (self.tbl_beads, self.btn_auto, self.btn_reverse, self.btn_clear_bead,
                  self.btn_clear_all, self.cmb_center, self.btn_est):
            w.setEnabled(on)

    def backmap_settings(self, random_spin=None) -> BackmapSettings:
        return BackmapSettings(
            scale=self.spn_scale.value(),
            mode=("rigid", "flex", "fragment")[self.cmb_mode.currentIndex()],
            flex_weight=self.spn_flex.value(),
            random_spin=self.chk_spin.isChecked() if random_spin is None else random_spin,
            seed=self.spn_seed.value(),
        )

    # ============================================================ file I/O
    def browse_cg(self):
        p, _ = QFileDialog.getOpenFileName(self, "CG LAMMPS data file", self._dir(),
                                           "LAMMPS data (*.data *.lmp *.dat *.txt);;All files (*)")
        if p:
            self.ed_cg.setText(p)
            self._remember(p)
            self.load_cg()

    def browse_aa(self):
        p, _ = QFileDialog.getOpenFileName(self, "All-atom moltemplate file", self._dir("last_aa_dir"),
                                           "moltemplate (*.lt);;All files (*)")
        if p:
            self.ed_aa.setText(p)
            self.ed_ff.clear()
            self._remember(p, "last_aa_dir")
            if self.cur is not None:
                self.load_aa()

    def browse_ff(self):
        p, _ = QFileDialog.getOpenFileName(self, "Force-field moltemplate file", self._dir("last_aa_dir"),
                                           "moltemplate (*.lt);;All files (*)")
        if p:
            self.ed_ff.setText(p)

    def browse_out(self):
        p = QFileDialog.getExistingDirectory(self, "Output folder", self.ed_out.text())
        if p:
            self.ed_out.setText(p)

    def browse_lmp(self):
        p, _ = QFileDialog.getOpenFileName(self, "LAMMPS executable", self.ed_lmp.text() or str(Path.home()))
        if p:
            self.ed_lmp.setText(p)

    # =========================================================== CG system
    def load_cg(self, keep_states: bool = False):
        path = self.ed_cg.text().strip()
        if not path:
            self.browse_cg()
            return
        style = self.cmb_style.currentText()
        split = {"auto": "auto", "molecule ID": "molid", "bonds": "bonds"}[self.cmb_split.currentText()]
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            data = read_lammps_data(path, style)
            cg = CGSystem(data, split)
        except Exception as exc:  # noqa: BLE001
            QApplication.restoreOverrideCursor()
            self.error("Cannot read CG data", str(exc))
            return
        QApplication.restoreOverrideCursor()
        self.cg = cg
        self._bond_len_cache.clear()
        if not keep_states:
            self.states = {}
        self.statusBar().showMessage("Select a species, then load its all-atom template", 8000)
        self.log(f"loaded {path}: {data.n_atoms} beads, atom style '{data.atom_style}', "
                 f"box {data.box.lengths.round(3).tolist()}, {len(cg.species)} species "
                 f"(molecules split by {cg.split_mode})")
        for sp in cg.species:
            self.log(f"  {sp.name}: {sp.count} molecules x {sp.n_beads} beads")
        self._fill_species_table()
        if cg.species:
            self.tbl_species.selectRow(0)

    def _fill_species_table(self):
        t = self.tbl_species
        t.blockSignals(True)
        t.setRowCount(0)
        for i, sp in enumerate(self.cg.species):
            t.insertRow(i)
            it = QTableWidgetItem(sp.name)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            st = self.states.get(i)
            it.setCheckState(Qt.Checked if (st is None or st.enabled) else Qt.Unchecked)
            it.setToolTip("Tick to include this species when back-mapping\n" + " ".join(sp.bead_names))
            t.setItem(i, 0, it)
            t.setItem(i, 1, QTableWidgetItem(str(sp.n_beads)))
            t.setItem(i, 2, QTableWidgetItem(str(sp.count)))
            t.setItem(i, 3, QTableWidgetItem(self._species_status(i)))
        t.resizeColumnsToContents()
        t.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        t.blockSignals(False)

    def _species_status(self, i: int) -> str:
        st = self.states.get(i)
        if st is None or st.aa is None:
            return "-"
        n = len(st.mapping.mapped_beads()) if st.mapping else 0
        if st.copies > 1:
            return f"{st.copies} x {st.aa.name} per bead"
        return f"{st.aa.name} ({n}/{st.mapping.n_beads} mapped)"

    def _update_species_row(self, i: int):
        if 0 <= i < self.tbl_species.rowCount():
            self.tbl_species.blockSignals(True)
            self.tbl_species.item(i, 3).setText(self._species_status(i))
            self.tbl_species.blockSignals(False)

    def on_species_item_changed(self, item: QTableWidgetItem):
        if item.column() == 0:
            self.states.setdefault(item.row(), SpeciesState()).enabled = item.checkState() == Qt.Checked

    def on_species_selected(self):
        rows = self.tbl_species.selectionModel().selectedRows()
        if not rows or self.cg is None:
            return
        self.cur = rows[0].row()
        st = self.state
        sp = self.species
        self.active_bead = 0
        self.ed_aa.setText(st.aa_path)
        self.ed_ff.setText(st.ff_path)
        self.spn_copies.blockSignals(True)
        self.spn_copies.setValue(st.copies)
        self.spn_copies.blockSignals(False)
        self.spn_instance.blockSignals(True)
        self.spn_instance.setMaximum(sp.count)
        self.spn_instance.setValue(1)
        self.spn_instance.blockSignals(False)
        self._set_species_widgets_enabled(True)
        self._update_aa_info()
        self.refresh_all(reset=True)

    def random_instance(self):
        if self.species:
            self.spn_instance.setValue(int(np.random.randint(1, self.species.count + 1)))

    # =========================================================== all-atom
    def load_aa(self):
        if self.cur is None:
            self.error("No species selected", "Load a CG system and select a species first.")
            return
        path = self.ed_aa.text().strip()
        if not path:
            self.browse_aa()
            return
        ff = self.ed_ff.text().strip() or None
        try:
            mol = parse_molecule(path, ff)
        except Exception as exc:  # noqa: BLE001
            self.error("Cannot read all-atom template", str(exc))
            return
        if mol.ff is None:
            QMessageBox.warning(self, "Force field not found",
                                "The force field this molecule inherits from was not found next to it.\n"
                                "Select the force-field .lt file manually (needed to write LAMMPS files).")
        st = self.state
        st.aa, st.aa_path = mol, path
        st.ff_path = ff or (mol.ff.path if mol.ff else "")
        self.ed_ff.setText(st.ff_path)
        sp = self.species
        st.mapping = BeadMapping(n_beads=sp.n_beads, center=self._center_mode())
        if sp.n_beads == 1:
            st.mapping.assign(0, np.flatnonzero(mol.heavy_mask()).tolist())
        self.active_bead = 0
        self.log(f"loaded all-atom template {mol.name} from {path}: {mol.n_atoms} atoms "
                 f"({int(mol.heavy_mask().sum())} heavy), total charge {mol.charges.sum():+.3f}, "
                 f"force field {mol.ff.name + ' (' + mol.ff.path + ')' if mol.ff else 'NOT FOUND'}")
        self._update_aa_info()
        self._update_species_row(self.cur)
        self._set_mapping_widgets_enabled(True)
        self.refresh_all(reset=True)

    def load_builtin_water(self):
        if self.cur is None:
            return
        ff_path = self.ed_ff.text().strip()
        if not ff_path:
            ff_path = next((st.ff_path for st in self.states.values() if st.ff_path), "")
        if not ff_path:
            self.error("Force field needed", "Select the force-field .lt file (step 2) or load another "
                                             "species' template first; water uses its OW/H atom types.")
            return
        try:
            mol = spc_water(parse_forcefield(ff_path))
        except Exception as exc:  # noqa: BLE001
            self.error("Cannot build SPC water", str(exc))
            return
        st, sp = self.state, self.species
        st.aa, st.aa_path, st.ff_path = mol, BUILTIN_SPC, ff_path
        st.mapping = BeadMapping(n_beads=sp.n_beads, center=self._center_mode())
        if sp.n_beads == 1:
            st.mapping.assign(0, [0])
        self.ed_aa.setText(BUILTIN_SPC)
        self.ed_ff.setText(ff_path)
        self.log(f"built-in SPC water (types OW/H of {mol.ff.name}) assigned to {sp.name}")
        self._update_aa_info()
        self._update_species_row(self.cur)
        self._set_mapping_widgets_enabled(True)
        self.refresh_all(reset=True)

    def on_copies_changed(self, v: int):
        if self.state is not None:
            self.state.copies = int(v)
            self._update_species_row(self.cur)
            self.refresh_cg(reset=False)

    def _update_aa_info(self):
        st = self.state
        if st is None or st.aa is None:
            self.lbl_aa_info.setText("")
            self.aa_view.set_info("")
            return
        m = st.aa
        ff = f"{m.ff.name}" if m.ff else "<span style='color:#c00'>not found</span>"
        self.lbl_aa_info.setText(f"{m.name}: {m.n_atoms} atoms, {int(m.heavy_mask().sum())} heavy, "
                                 f"q = {m.charges.sum():+.3f} e<br>Force field: {ff}")
        self.aa_view.set_info(f"{m.name}   {m.n_atoms} atoms   {int(m.heavy_mask().sum())} heavy")

    # ============================================================ mapping
    def _center_mode(self) -> str:
        return ["com", "heavy_com", "centroid"][self.cmb_center.currentIndex()]

    def on_center_changed(self):
        st = self.state
        if st and st.mapping:
            st.mapping.center = self._center_mode()
            self.refresh_cg(reset=False)

    def _edit_mapping(self, fn):
        st = self.state
        if st is None or st.mapping is None:
            return
        fn(st.mapping)
        self._update_species_row(self.cur)
        self.refresh_all(reset=False)

    def auto_map(self):
        st, sp = self.state, self.species
        if st is None or st.aa is None:
            return
        try:
            m = auto_linear_mapping(st.aa, sp.n_beads)
        except Exception as exc:  # noqa: BLE001
            self.error("Auto mapping failed", str(exc))
            return
        m.center = self._center_mode()
        st.mapping = m
        self.log("auto mapping: " + " | ".join(
            f"{k + 1}:{','.join(st.aa.atom_names[a] for a in b)}" for k, b in enumerate(m.beads)))
        self._update_species_row(self.cur)
        self.refresh_all(reset=False)

    def reverse_map(self):
        st = self.state
        if st and st.mapping:
            st.mapping = reverse_mapping(st.mapping)
            self._update_species_row(self.cur)
            self.refresh_all(reset=False)

    def set_active_bead(self, k: int):
        sp = self.species
        if sp is None or k >= sp.n_beads:
            return
        self.active_bead = k
        self.refresh_all(reset=False)

    def on_bead_row_selected(self):
        rows = self.tbl_beads.selectionModel().selectedRows()
        if rows and rows[0].row() != self.active_bead:
            self.set_active_bead(rows[0].row())

    def on_aa_clicked(self, i: int):
        st = self.state
        if st is None or st.mapping is None:
            return
        st.mapping.toggle(self.active_bead, i)
        self._update_species_row(self.cur)
        self.refresh_all(reset=False)

    def on_aa_box(self, atoms: list):
        st = self.state
        if st is None or st.mapping is None:
            return
        st.mapping.assign(self.active_bead, atoms)
        self._update_species_row(self.cur)
        self.refresh_all(reset=False)

    def on_aa_hover(self, i: int):
        st = self.state
        if i < 0 or st is None or st.aa is None:
            self.statusBar().clearMessage()
            return
        m = st.aa
        k = st.mapping.bead_of(i) if st.mapping else -1
        owner = atom_owners(m, st.mapping)[i] if st.mapping else -1
        where = (f"bead {k + 1} ({self.species.bead_names[k]})" if k >= 0 else
                 (f"follows bead {owner + 1}" if owner >= 0 else "unassigned"))
        self.statusBar().showMessage(f"{m.atom_names[i]}  type {m.atom_types[i]}  q={m.charges[i]:+.3f}  "
                                     f"{m.elements[i]}  -> {where}")

    def on_cg_hover(self, k: int):
        sp = self.species
        if k < 0 or sp is None:
            self.statusBar().clearMessage()
            return
        st = self.state
        atoms = ""
        if st and st.aa and st.mapping:
            atoms = ", ".join(st.aa.atom_names[a] for a in st.mapping.beads[k]) or "no atoms"
        self.statusBar().showMessage(f"bead {k + 1}: {sp.bead_names[k]}  (type {sp.bead_types[k]})  {atoms}")

    # ============================================================ drawing
    def refresh_all(self, reset: bool):
        self.refresh_bead_table()
        self.refresh_aa(reset)
        self.refresh_cg(reset)
        st = self.state
        if st and st.aa and st.mapping:
            msgs = st.mapping.validate(st.aa)
            self.lbl_valid.setText("<br>".join(f"&bull; {m}" for m in msgs) if msgs
                                   else "<span style='color:#2a7'>All heavy atoms mapped.</span>")
        else:
            self.lbl_valid.setText("")

    def refresh_bead_table(self):
        sp, st = self.species, self.state
        t = self.tbl_beads
        t.blockSignals(True)
        t.setRowCount(0)
        if sp is not None:
            for k in range(sp.n_beads):
                t.insertRow(k)
                it = QTableWidgetItem(f"{k + 1}")
                it.setIcon(_swatch(BEAD_COLORS[k % len(BEAD_COLORS)]))
                t.setItem(k, 0, it)
                t.setItem(k, 1, QTableWidgetItem(sp.bead_names[k]))
                names = ""
                if st and st.aa and st.mapping:
                    names = " ".join(st.aa.atom_names[a] for a in st.mapping.beads[k])
                t.setItem(k, 2, QTableWidgetItem(names))
            t.selectRow(self.active_bead)
        t.blockSignals(False)

    def _aa_colors(self, mol: AAMolecule, mapping: BeadMapping | None) -> list[QColor]:
        by_elem = [QColor(ELEMENT_COLORS.get(e, "#b0b0b0")) for e in mol.elements]
        if self.cmb_color.currentIndex() == 1 or mapping is None:
            return by_elem
        owner = atom_owners(mol, mapping)
        cols = []
        for i, e in enumerate(mol.elements):
            k = owner[i]
            if k < 0:
                cols.append(by_elem[i] if e != "H" else QColor("#f4f4f4"))
                continue
            c = QColor(BEAD_COLORS[k % len(BEAD_COLORS)])
            if e == "H":
                c = QColor.fromRgbF(0.45 * c.redF() + 0.55, 0.45 * c.greenF() + 0.55, 0.45 * c.blueF() + 0.55)
            elif mapping.bead_of(i) < 0:
                c = by_elem[i]
            cols.append(c)
        return cols

    def refresh_aa(self, reset: bool):
        st = self.state
        if st is None or st.aa is None:
            self.aa_view.clear(self.aa_view.placeholder)
            return
        m = st.aa
        heavy = m.heavy_mask()
        radii = np.where(heavy, 0.42, 0.24)
        if reset or len(self.aa_view.pos) != m.n_atoms:
            self.aa_view.set_molecule(m.pos, radii, self._aa_colors(m, st.mapping), m.bonds,
                                      [f"{n}" for n in m.atom_names], pickable=heavy, reset_view=False)
            self.aa_view.set_visible(heavy | self.chk_h.isChecked())
            self.aa_view.reset_view()
        else:
            self.aa_view.set_colors(self._aa_colors(m, st.mapping))
            self.aa_view.set_visible(heavy | self.chk_h.isChecked())
        if st.mapping:
            ring = QColor("#ffd400")
            self.aa_view.set_rings({a: ring for a in st.mapping.beads[self.active_bead]})

    def _bond_len(self) -> float:
        if self.cur not in self._bond_len_cache:
            self._bond_len_cache[self.cur] = self.cg.mean_bond_length(self.species, 100)
        return self._bond_len_cache[self.cur]

    def refresh_cg(self, reset: bool):
        sp = self.species
        if sp is None:
            self.cg_view.clear(self.cg_view.placeholder)
            return
        k = self.spn_instance.value() - 1
        s = self.spn_scale.value()
        x = self.cg.instance_coords(sp, k)
        bl = self._bond_len()
        r = 0.32 * (bl if np.isfinite(bl) else 0.5)   # view works in DPD length units
        st = self.state
        overlay = bool(self.chk_overlay.isChecked() and st and st.aa and st.mapping
                       and (len(st.mapping.mapped_beads()) >= 2 or st.copies > 1))
        cols = []
        for b in range(sp.n_beads):
            c = QColor(BEAD_COLORS[b % len(BEAD_COLORS)])
            if overlay:
                c.setAlpha(90)
            cols.append(c)
        labels = [f"{b + 1}:{n}" for b, n in enumerate(sp.bead_names)]
        if reset or len(self.cg_view.pos) != sp.n_beads:
            self.cg_view.set_molecule(x, np.full(sp.n_beads, r), cols, sp.bonds, labels, reset_view=True)
        else:
            self.cg_view.pos = x
            self.cg_view.radii = np.full(sp.n_beads, r)
            self.cg_view.set_colors(cols)
        self.cg_view.set_rings({self.active_bead: QColor("#ffd400")})
        if overlay:
            try:
                cols = self._aa_colors(st.aa, st.mapping)
                if st.copies > 1 and sp.n_beads == 1:
                    ys = place_cluster(x[0] * s, st.aa, st.copies, np.random.default_rng(k))
                    pts, pc, pb = [], [], []
                    for y in ys:
                        off = len(pts)
                        pts += list(y / s)
                        pc += [QColor(ELEMENT_COLORS.get(e, "#b0b0b0")) for e in st.aa.elements]
                        pb += [(off + a, off + b) for a, b in st.aa.bonds]
                    self.cg_view.set_overlay(np.array(pts), pc, pb, radius=0.3 / s)
                    self.cg_view.set_info(f"{sp.name}   bead {k + 1}/{sp.count}   "
                                          f"{st.copies} x {st.aa.name}")
                else:
                    fitter = Fitter(st.aa, st.mapping, self.backmap_settings(random_spin=False), sp.bonds)
                    y = fitter.fit(x, np.random.default_rng(0))
                    heavy = np.flatnonzero(st.aa.heavy_mask())
                    idx = {a: j for j, a in enumerate(heavy)}
                    hb = [(idx[a], idx[b]) for a, b in st.aa.bonds if a in idx and b in idx]
                    self.cg_view.set_overlay(y[heavy] / s, [cols[a] for a in heavy], hb, radius=0.35 / s)
                    rmsd = fitter.rmsd(x, y)
                    self.cg_view.set_info(f"{sp.name}   molecule {k + 1}/{sp.count}   "
                                          f"bead-fit RMSD {rmsd:.2f} A")
            except Exception as exc:  # noqa: BLE001
                self.cg_view.set_overlay(np.zeros((0, 3)), [], [])
                self.statusBar().showMessage(f"overlay: {exc}")
        else:
            self.cg_view.set_overlay(np.zeros((0, 3)), [], [])
            self.cg_view.set_info(f"{sp.name}   molecule {k + 1}/{sp.count}")

    # ============================================================ actions
    def estimate_scale(self):
        st, sp = self.state, self.species
        if not (st and st.aa and st.mapping):
            return
        try:
            s = estimate_scale(self.cg, sp, st.aa, st.mapping)
        except Exception as exc:  # noqa: BLE001
            self.error("Cannot estimate scale", str(exc))
            return
        self.spn_scale.setValue(s)
        self.log(f"estimated scale for {sp.name}: {s:.3f} A per DPD length unit "
                 f"(template bead distances / CG bond lengths)")

    def output_settings(self) -> OutputSettings:
        for txt, what in ((self.ed_k.text(), "restraint constants"), (self.ed_dt.text(), "time steps")):
            try:
                [float(v) for v in txt.split()]
            except ValueError:
                raise ValueError(f"{what} must be numbers separated by spaces: {txt!r}") from None
        return OutputSettings(
            basename=self.ed_base.text().strip() or "system", cutoff=self.spn_cut.value(),
            long_range=self.chk_long.isChecked(), soft_stage=self.chk_soft.isChecked(),
            min_steps=self.spn_steps.value(), bonded_stage=self.chk_bonded.isChecked(),
            restrained=self.chk_restr.isChecked() and bool(self.ed_k.text().split()),
            restraint_k=self.ed_k.text().strip() or "0", md_steps=self.spn_md.value(),
            md_timesteps=self.ed_dt.text().strip() or "1.0", temperature=self.spn_temp.value(),
            release=self.chk_release.isChecked())

    def _jobs(self):
        jobs, skipped = [], []
        for i, sp in enumerate(self.cg.species):
            st = self.states.get(i)
            if st is not None and not st.enabled:
                continue
            if st is None or st.aa is None or st.mapping is None or not st.mapping.mapped_beads():
                skipped.append(sp.name)
                continue
            jobs.append(Job(sp, st.aa, st.mapping, st.copies if sp.n_beads == 1 else 1))
        return jobs, skipped

    def run(self):
        if self.cg is None:
            self.error("Nothing to do", "Load a CG system first.")
            return
        jobs, skipped = self._jobs()
        if not jobs:
            self.error("Nothing to do", "No species has an all-atom template with a mapping yet.")
            return
        for j in jobs:
            if j.mol.ff is None:
                self.error("Missing force field", f"{j.mol.name} has no force field; select it in step 2.")
                return
        if skipped:
            self.log("species without template/mapping are left out: " + ", ".join(skipped))
        bm = self.backmap_settings()
        ov = OverlapSettings(enabled=self.g_ov.isChecked(), d_min=self.spn_dmin.value(),
                             max_iter=self.spn_oviter.value(), heavy_only=self.chk_heavy.isChecked())
        try:
            outs = self.output_settings()
        except ValueError as exc:
            self.error("Invalid relaxation settings", str(exc))
            return
        mini = MinimizeSettings(enabled=self.g_min.isChecked(), lammps_exe=self.ed_lmp.text().strip(),
                                mpi=self.spn_mpi.value())
        if mini.enabled:
            self.settings.setValue("lammps_exe", mini.lammps_exe)
        half = 0.5 * float((self.cg.box.lengths * bm.scale).min())
        if outs.cutoff >= half:
            self.error("Cutoff too large", f"Pair cutoff {outs.cutoff} A must be smaller than half the "
                                           f"scaled box ({half:.1f} A).")
            return
        out_dir = self.ed_out.text().strip()
        cg = self.cg

        def task(log, progress, stop):
            return run_backmapping(cg, jobs, bm, ov, outs, out_dir, mini, log=log,
                                   progress=progress, stop=stop)

        self.thread = QThread(self)
        self.worker = Worker(task)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self.log)
        self.worker.progress.connect(self.on_progress)
        self.worker.done.connect(self.on_done)
        self.worker.failed.connect(self.on_failed)
        self.worker.done.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(lambda: self._busy(False))
        self._busy(True)
        self.log("=" * 60)
        self.log(f"back-mapping: scale {bm.scale:.3f} A/unit, fit {bm.mode}, output {out_dir}")
        self.thread.start()

    def stop(self):
        if self.worker:
            self.worker.stop_requested = True
            self.log("stop requested (takes effect at the next LAMMPS output line)")

    def _busy(self, on: bool):
        self.btn_run.setEnabled(not on)
        self.btn_stop.setEnabled(on)
        self.progress.setRange(0, 0 if on else 1)
        if not on:
            self.progress.setValue(1)

    def on_progress(self, name: str, i: int, n: int):
        self.progress.setRange(0, n)
        self.progress.setValue(i)
        self.progress.setFormat("fitting %v / %m")

    def on_done(self, res):
        self.progress.setFormat("done")
        wr = res.write
        msg = (f"Wrote {wr.n_atoms} atoms in {wr.n_molecules} molecules.\n\n"
               + "\n".join(f"{k}: {v}" for k, v in wr.files.items()))
        if wr.warnings:
            msg += "\n\nWarnings:\n" + "\n".join(wr.warnings)
        if res.lammps_exit is not None:
            msg += f"\n\nLAMMPS exit code: {res.lammps_exit}"
            if res.lammps_exit == 0:
                msg += f"\nMinimised structure: {Path(self.ed_out.text()) / (self.ed_base.text() + '_min.data')}"
            else:
                msg += "\nSee log.lammps in the output folder."
        QMessageBox.information(self, "Back-mapping finished", msg)

    def on_failed(self, msg: str):
        self.progress.setFormat("failed")
        self.error("Back-mapping failed", msg)

    # ============================================================ project
    def current_project(self) -> Project:
        p = Project(cg_path=self.ed_cg.text().strip(), atom_style=self.cmb_style.currentText(),
                    split={"auto": "auto", "molecule ID": "molid", "bonds": "bonds"}[self.cmb_split.currentText()],
                    out_dir=self.ed_out.text().strip())
        if self.cg:
            for i, st in sorted(self.states.items()):
                if st.aa is None or st.mapping is None:
                    continue
                sp = self.cg.species[i]
                p.assignments.append(SpeciesAssignment(
                    species_bead_names=sp.bead_names, species_index=i, aa_path=st.aa_path,
                    ff_path=st.ff_path or None, mapping=st.mapping.to_dict(st.aa), enabled=st.enabled,
                    copies_per_bead=st.copies))
        p.backmap = self.backmap_settings()
        p.overlap = OverlapSettings(enabled=self.g_ov.isChecked(), d_min=self.spn_dmin.value(),
                                    max_iter=self.spn_oviter.value(), heavy_only=self.chk_heavy.isChecked())
        p.output = self.output_settings()
        p.minimize = MinimizeSettings(enabled=self.g_min.isChecked(), lammps_exe=self.ed_lmp.text().strip(),
                                      mpi=self.spn_mpi.value())
        return p

    def save_project(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save project", self._dir("last_proj_dir"),
                                           "revdpd project (*.json)")
        if not p:
            return
        if not p.endswith(".json"):
            p += ".json"
        self.current_project().save(p)
        self._remember(p, "last_proj_dir")
        self.log(f"project saved to {p}")

    def load_project(self, path: str | None = None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Load project", self._dir("last_proj_dir"),
                                                  "revdpd project (*.json)")
        if not path:
            return
        try:
            p = Project.load(path)
        except Exception as exc:  # noqa: BLE001
            self.error("Cannot load project", str(exc))
            return
        self._remember(path, "last_proj_dir")
        self.apply_project(p)
        self.log(f"project loaded from {path}")

    def apply_project(self, p: Project):
        self.ed_cg.setText(p.cg_path)
        self.cmb_style.setCurrentText(p.atom_style)
        self.cmb_split.setCurrentText({"auto": "auto", "molid": "molecule ID", "bonds": "bonds"}[p.split])
        b = p.backmap
        self.spn_scale.setValue(b.scale)
        self.cmb_mode.setCurrentIndex({"rigid": 0, "flex": 1, "fragment": 2}.get(b.mode, 0))
        self.spn_flex.setValue(b.flex_weight)
        self.chk_spin.setChecked(b.random_spin)
        self.spn_seed.setValue(b.seed)
        self.g_ov.setChecked(p.overlap.enabled)
        self.spn_dmin.setValue(p.overlap.d_min)
        self.spn_oviter.setValue(p.overlap.max_iter)
        self.chk_heavy.setChecked(p.overlap.heavy_only)
        self.ed_out.setText(p.out_dir)
        self.ed_base.setText(p.output.basename)
        self.spn_cut.setValue(p.output.cutoff)
        self.chk_long.setChecked(p.output.long_range)
        o = p.output
        self.chk_soft.setChecked(o.soft_stage)
        self.spn_steps.setValue(o.min_steps)
        self.chk_bonded.setChecked(o.bonded_stage)
        self.chk_restr.setChecked(o.restrained)
        self.ed_k.setText(o.restraint_k)
        self.spn_md.setValue(o.md_steps)
        self.ed_dt.setText(o.md_timesteps)
        self.spn_temp.setValue(o.temperature)
        self.chk_release.setChecked(o.release)
        self.g_min.setChecked(p.minimize.enabled)
        if p.minimize.lammps_exe:
            self.ed_lmp.setText(p.minimize.lammps_exe)
        self.spn_mpi.setValue(p.minimize.mpi)
        self.load_cg()
        if self.cg is None:
            return
        for a in p.assignments:
            try:
                sp = resolve_species(self.cg, a)
                i = self.cg.species.index(sp)
                mol = load_template(a.aa_path, a.ff_path)
                mp = BeadMapping.from_dict(a.mapping, mol)
            except Exception as exc:  # noqa: BLE001
                self.error("Cannot restore species", str(exc))
                continue
            self.states[i] = SpeciesState(aa=mol, mapping=mp, aa_path=a.aa_path,
                                          ff_path=a.ff_path or (mol.ff.path if mol.ff else ""),
                                          enabled=a.enabled, copies=a.copies_per_bead)
        self._fill_species_table()
        if self.cg.species:
            first = min((self.cg.species.index(resolve_species(self.cg, a)) for a in p.assignments), default=0)
            self.tbl_species.selectRow(first)
            self.on_species_selected()
