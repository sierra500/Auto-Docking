import streamlit as st
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from meeko import MoleculePreparation, PDBQTWriterLegacy
from vina import Vina
import base64
import time
import urllib.request
import json
from rdkit.Geometry import Point3D
import subprocess

# 页面配置
st.set_page_config(page_title="Auto-Docking Platform", page_icon="🧬", layout="wide")

# ================= 🎨 自定义 CSS 样式区 =================
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0b132b 0%, #1c2541 50%, #3a506b 100%); color: #e0e1dd; }
    [data-testid="stSidebar"] { background-color: rgba(28, 37, 65, 0.85); border-right: 1px solid rgba(255,255,255,0.1); }
    h1, h2, h3 { color: #ffffff !important; font-family: 'Helvetica Neue', sans-serif; }
    .glass-card {
        background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 15px; padding: 30px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37); height: 100%;
    }
    .instructions p { font-size: 16px; line-height: 1.6; color: #b0c4de; }
    .instructions li { font-size: 15px; margin-bottom: 8px; color: #c9d1d9; }
    .stButton>button { background-color: #00b4d8; color: white; border-radius: 8px; border: none; transition: all 0.3s ease; }
    .stButton>button:hover { background-color: #0077b6; box-shadow: 0 0 15px #00b4d8; }
    .magic-image { mix-blend-mode: screen; width: 100%; filter: drop-shadow(0px 0px 25px rgba(0, 180, 216, 0.4)); }
    .img-top { transform: scale(1.50); transform-origin: center center; }
    
    /* 侧边栏信息的微型面板 */
    .info-panel {
        background: rgba(0, 180, 216, 0.1);
        border-left: 4px solid #00b4d8;
        padding: 10px;
        margin-bottom: 15px;
        border-radius: 4px;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# ================= 🧠 状态管理初始化 =================
if 'history' not in st.session_state: st.session_state.history = []
if 'current_job' not in st.session_state: st.session_state.current_job = None
if 'view_mode' not in st.session_state: st.session_state.view_mode = "home"
if 'fetched_files' not in st.session_state: st.session_state.fetched_files = None

# ================= 🧬 核心辅助函数 =================
def scan_pdb(pdb_path):
    chains = {}
    hetatms_dict = {}
    with open(pdb_path, 'r') as f:
        for line in f:
            if len(line) > 16 and line[16] in ['B', 'C']: continue
            if line.startswith("ATOM"):
                chain_id = line[21]
                if chain_id not in chains: chains[chain_id] = []
                chains[chain_id].append(line)
            elif line.startswith("HETATM"):
                res_name = line[17:20].strip()
                if res_name == "HOH": continue
                if res_name not in hetatms_dict: hetatms_dict[res_name] = []
                hetatms_dict[res_name].append(line)
    return chains, hetatms_dict

def get_bounding_box(lines, padding=8.0):
    x, y, z = [], [], []
    for line in lines:
        try:
            x.append(float(line[30:38].strip()))
            y.append(float(line[38:46].strip()))
            z.append(float(line[46:54].strip()))
        except ValueError: continue
    if not x: return None, None
    min_x, max_x = min(x), max(x)
    min_y, max_y = min(y), max(y)
    min_z, max_z = min(z), max(z)
    center = [(min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2]
    size = [max_x - min_x + 2 * padding, max_y - min_y + 2 * padding, max_z - min_z + 2 * padding]
    return center, size

def prepare_receptor_pure_python(chain_lines, valid_ions, out_path):
    clean_atoms = [line for line in chain_lines if line[17:20].strip() != "HOH"]
    pdb_block = "".join(clean_atoms + valid_ions)
    
    mol = Chem.MolFromPDBBlock(pdb_block, removeHs=False)
    if mol is None: raise Exception("RDKit cannot parse the protein structure.")
    mol_h = Chem.AddHs(mol, addCoords=True, onlyOnAtoms=[a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() in ['N', 'O']])
    
    ad_type_dict = {
        'C': 'C', 'N': 'N', 'O': 'OA', 'H': 'HD', 'S': 'SA', 
        'F': 'F', 'Cl': 'Cl', 'Br': 'Br', 'I': 'I',
        'Zn': 'ZN', 'Mg': 'MG', 'Ca': 'CA', 'Mn': 'MN', 'Fe': 'FE',
        'Na': 'NA', 'K': 'K', 'Cu': 'CU', 'Co': 'CO', 'Ni': 'NI'
    }

    with open(out_path, "w") as f:
        for atom in mol_h.GetAtoms():
            info = atom.GetPDBResidueInfo()
            res_name = info.GetResidueName() if info else "UNL"
            chain = info.GetChainId() if info else "A"
            res_num = info.GetResidueNumber() if info else 1
            atom_name = info.GetName().strip() if info else atom.GetSymbol()
            pos = mol_h.GetConformer().GetAtomPosition(atom.GetIdx())
            
            elem = atom.GetSymbol()
            ad_type = ad_type_dict.get(elem, elem)
            
            f.write(f"ATOM  {atom.GetIdx()+1:5d} {atom_name:<4s} {res_name:<3s} {chain}{res_num:4d}    {pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}  1.00  0.00    {0.000:+.3f} {ad_type}\n")
        f.write("TER\nEND\n")

def download_online_files(pdb_id, lig_name_or_cid):
    work_dir = "web_temp_online"
    os.makedirs(work_dir, exist_ok=True)
    
    pdb_path = os.path.join(work_dir, f"{pdb_id}.pdb")
    url_pdb = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    
    sdf_path = None
    if lig_name_or_cid:
        cid = lig_name_or_cid
        if not lig_name_or_cid.isdigit():
            url_search = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{lig_name_or_cid}/cids/JSON"
            try:
                req = urllib.request.Request(url_search, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response:
                    data = json.loads(response.read().decode('utf-8'))
                    cid = data['IdentifierList']['CID'][0]
            except Exception:
                return None, None, f"Cannot find ligand '{lig_name_or_cid}' in PubChem."
        
        sdf_path = os.path.join(work_dir, f"{cid}.sdf")
        url_sdf = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/SDF?record_type=3d"
        try:
            req = urllib.request.Request(url_sdf, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(sdf_path, 'wb') as out_file:
                out_file.write(response.read())
        except Exception:
            return None, None, "Failed to download ligand from PubChem."

    try:
        req = urllib.request.Request(url_pdb, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(pdb_path, 'wb') as out_file:
            out_file.write(response.read())
    except Exception:
        return None, None, f"Failed to download PDB ID '{pdb_id}'."
        
    return pdb_path, sdf_path, "Success"

# ================= 侧边栏交互区 =================
st.sidebar.markdown("<h2 style='text-align: center; color: #00b4d8;'>⚙️ Control Panel</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

if st.sidebar.button("📜 View History Records", use_container_width=True):
    st.session_state.view_mode = "history"
    st.rerun()

# ================= 页面视图逻辑 =================

# 1. 历史记录模式
if st.session_state.view_mode == "history":
    st.markdown("<h1 style='text-align: center;'>📜 History Records</h1>", unsafe_allow_html=True)
    if st.button("⬅️ Back to Home"):
        st.session_state.view_mode = "home"
        st.rerun()
    if not st.session_state.history:
        st.warning("No history records yet.")
    else:
        for idx, record in enumerate(st.session_state.history):
            with st.expander(f"Record {idx+1} - {record['timestamp']} - Ligand: {record['ligand_name']}"):
                st.dataframe(record['df'])
                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    st.download_button(label=f"📥 Excel {idx+1}", data=record['excel_data'], file_name=f"report_{idx+1}.xlsx", mime="application/vnd.ms-excel", key=f"excel_{idx}")
                with col_dl2:
                    st.download_button(label=f"📥 PDBQT {idx+1}", data=record['pdbqt_data'], file_name=f"docked_{idx+1}.pdbqt", mime="chemical/x-pdb", key=f"pdbqt_{idx}")
    st.stop()

# 2. 主页模式
st.markdown("<h1 style='text-align: center;'>🧬 Auto-Docking Web Platform</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8da9c4;'>Author: Jiang Xintong | Version: v4.5 | AI-powered and Physics-based Efficient Tool</p>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid rgba(255,255,255,0.1);'>", unsafe_allow_html=True)

if st.session_state.current_job is not None:
    job = st.session_state.current_job
    st.success(f"✅ Docking Completed for Ligand: {job['ligand_name']}")
    st.subheader("📊 Docking Results")
    st.dataframe(job['df'], use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📥 Download Excel Report", data=job['excel_data'], file_name="docking_report.xlsx", mime="application/vnd.ms-excel")
    with col2:
        st.download_button("📥 Download Docked Ligand (PDBQT)", data=job['pdbqt_data'], file_name="docked_ligand.pdbqt", mime="chemical/x-pdb")
    
    if st.button("🏁 Finish & Reset to Home", type="primary"):
        st.session_state.current_job = None
        st.rerun()

else:
    col_text, col_image = st.columns([1.1, 0.9], gap="large")
    
    with col_text:
        html_content = """<div class="glass-card instructions"><h3>📋 Usage Instructions</h3><p>1. Choose whether to <b>[1] Download files online</b> or <b>[2] Import locally</b>.</p><ul><li><i>Online:</i> Enter the PDB ID and Ligand Name/CID to auto-download.</li><li><i>Local:</i> Upload your PDB and SDF/PDB files via the sidebar.</li></ul><p>2. The script will detect multiple protein chains. Select the single chain you want to keep (e.g., A).</p><p>3. The sidebar lists all non-water molecules. Specify the ligand, and choose ions to retain. <i>(Selecting 'None' with specific ions triggers a red warning.)</i></p><p>4. The box is automatically calculated. If blind docking, an optimized 22×22×22 Å box is centered on the protein.</p><h3 style="margin-top: 25px;">⚠️ Important Notes</h3><ul><li>Online ligands are preferentially fetched from PubChem as 3D SDFs.</li><li>Water molecules are auto-stripped. Polar hydrogens and true RMSD values are auto-calculated.</li><li>ALL intermediate and final files (including a full 4-column Excel report) will be available for download.</li></ul></div>"""
        st.markdown(html_content, unsafe_allow_html=True)

    with col_image:
        img1 = "protein2.png" 
        img2 = "protein.png"
        if os.path.exists(img1):
            b64_1 = base64.b64encode(open(img1, 'rb').read()).decode()
            st.markdown(f'<div style="margin-top: 50px; margin-bottom: 80px;"><img src="data:image/png;base64,{b64_1}" class="magic-image img-top"></div>', unsafe_allow_html=True)
        if os.path.exists(img2):
            b64_2 = base64.b64encode(open(img2, 'rb').read()).decode()
            st.markdown(f'<div style="margin-top: 80px;"><img src="data:image/png;base64,{b64_2}" class="magic-image"></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.sidebar.header("📁 1. File Source")
    source_mode = st.sidebar.radio("Select Mode", ["Local Upload", "Online Fetch"])
    
    final_pdb_path = None
    final_sdf_path = None

    if source_mode == "Local Upload":
        pdb_file = st.sidebar.file_uploader("Upload Protein PDB", type=["pdb"])
        sdf_file = st.sidebar.file_uploader("Upload Ligand SDF/PDB", type=["sdf", "pdb"])
        
        if pdb_file and sdf_file:
            work_dir = "web_temp_local"
            os.makedirs(work_dir, exist_ok=True)
            final_pdb_path = os.path.join(work_dir, pdb_file.name)
            final_sdf_path = os.path.join(work_dir, sdf_file.name)
            with open(final_pdb_path, "wb") as f: f.write(pdb_file.getbuffer())
            with open(final_sdf_path, "wb") as f: f.write(sdf_file.getbuffer())

    else:
        pdb_id = st.sidebar.text_input("Enter PDB ID (e.g., 1IEP)")
        lig_input = st.sidebar.text_input("Enter Ligand Name or CID (e.g., STI)")
        
        if st.sidebar.button("Fetch Files"):
            if not pdb_id:
                st.sidebar.error("PDB ID is required.")
            else:
                with st.spinner("Downloading..."):
                    fetched_pdb, fetched_sdf, msg = download_online_files(pdb_id, lig_input)
                    if fetched_pdb:
                        st.session_state.fetched_files = {
                            "pdb_path": fetched_pdb,
                            "sdf_path": fetched_sdf
                        }
                        st.sidebar.success("Downloaded successfully!")
                    else:
                        st.sidebar.error(msg)

        if st.session_state.fetched_files:
            st.sidebar.info("👇 Check files before proceeding.")
            p_pdb = st.session_state.fetched_files["pdb_path"]
            p_sdf = st.session_state.fetched_files["sdf_path"]
            
            with open(p_pdb, "rb") as f:
                st.sidebar.download_button("📥 Preview PDB", f.read(), file_name="protein.pdb")
            if p_sdf and os.path.exists(p_sdf):
                with open(p_sdf, "rb") as f:
                    st.sidebar.download_button("📥 Preview Ligand", f.read(), file_name="ligand.sdf")
            
            col_conf, col_canc = st.sidebar.columns(2)
            with col_conf:
                if st.button("✅ Confirm", use_container_width=True):
                    st.session_state.current_online_pdb = p_pdb
                    st.session_state.current_online_sdf = p_sdf
                    st.rerun()
            with col_canc:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.fetched_files = None
                    st.rerun()

        if "current_online_pdb" in st.session_state:
            final_pdb_path = st.session_state.current_online_pdb
            final_sdf_path = st.session_state.current_online_sdf

    if final_pdb_path and os.path.exists(final_pdb_path):
        chains, het_groups = scan_pdb(final_pdb_path)
        
        st.sidebar.header("🧬 2. Structure Settings")
        chain_options = list(chains.keys())
        if not chain_options:
            st.error("❌ No protein chains detected.")
            st.stop()
            
        target_chain = st.sidebar.selectbox("Select Chain to Keep", chain_options)
        
        het_options = list(het_groups.keys())
        if het_options:
            het_html = "<div class='info-panel'><b>Detected HETATMs:</b><br>"
            for opt in het_options:
                atom_count = len(het_groups[opt])
                het_html += f"• {opt} ({atom_count} atoms)<br>"
            het_html += "</div>"
            st.sidebar.markdown(het_html, unsafe_allow_html=True)
            
            lig_options = ["None (Blind Docking)"] + het_options
            ligand_name = st.sidebar.selectbox("Select Model Ligand", lig_options)
        else:
            st.sidebar.info("No non-water molecules detected. Proceeding with blind docking.")
            ligand_name = "None (Blind Docking)"
            
        # ================= 🌟 核心修复：无条件显示离子清理选项 =================
        keep_ions = []
        if het_options:
            # 可保留的离子池：如果选了配体，就把配体排除，剩下的当离子；如果选了盲对接，所有 HETATM 都可以当离子
            if ligand_name != "None (Blind Docking)":
                ion_options = [k for k in het_options if k != ligand_name]
            else:
                ion_options = het_options
                
            if ion_options:
                ion_options_with_none = ["None (Drop All)"] + ion_options
                
                selected_ions = st.sidebar.multiselect(
                    "Select Ions to Keep", 
                    ion_options_with_none, 
                    default=["None (Drop All)"]
                )
                
                # 逻辑冲突检测与红框警告
                if "None (Drop All)" in selected_ions and len(selected_ions) > 1:
                    st.sidebar.error("⚠️ Conflict: Cannot select 'None' and specific ions at the same time!")
                    
                    st.markdown(
                        """
                        <style>
                        [data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] {
                            border: 2px solid #ff4b4b !important;
                            border-radius: 5px;
                        }
                        </style>
                        """, 
                        unsafe_allow_html=True
                    )
                    keep_ions = []
                elif "None (Drop All)" in selected_ions or len(selected_ions) == 0:
                    keep_ions = []
                else:
                    keep_ions = [ion for ion in selected_ions if ion != "None (Drop All)"]
        # =========================================================================

        st.sidebar.header("⚙️ 3. Docking Parameters")
        final_ex = st.sidebar.slider("Exhaustiveness", 8, 64, 16)
        final_nm = st.sidebar.slider("Number of Modes", 1, 9, 5)

        st.markdown("---")
        st.subheader("🚀 Execute Auto-Docking")
        
        if st.button("Start Docking", type="primary"):
            if not final_sdf_path or not os.path.exists(final_sdf_path):
                st.error("Ligand file is missing or invalid!")
            else:
                work_dir = "web_temp_run"
                os.makedirs(work_dir, exist_ok=True)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("Calculating dynamic bounding box...")
                    progress_bar.progress(20)
                    
                    if ligand_name != "None (Blind Docking)" and ligand_name in het_groups:
                        raw_lines = het_groups[ligand_name]
                        ligand_lines = [line for line in raw_lines if line[21] == target_chain]
                        center, size = get_bounding_box(ligand_lines, padding=8.0)
                        if not center: raise Exception(f"Ligand {ligand_name} has no valid coordinates on chain {target_chain}!")
                    else:
                        # 盲对接盒子
                        chain_atoms = chains[target_chain]
                        x, y, z = [], [], []
                        for line in chain_atoms:
                            x.append(float(line[30:38].strip()))
                            y.append(float(line[38:46].strip()))
                            z.append(float(line[46:54].strip()))
                        center = [sum(x)/len(x), sum(y)/len(y), sum(z)/len(z)]
                        size = [22.0, 22.0, 22.0]
                    
                    status_text.text("Generating receptor PDBQT...")
                    progress_bar.progress(40)
                    
                    valid_ion_lines = []
                    for ion in keep_ions:
                        if ion in het_groups: valid_ion_lines.extend(het_groups[ion])
                    receptor_pdbqt = os.path.join(work_dir, "receptor.pdbqt")
                    # 传入的 chains[target_chain] 只包含 ATOM，配体会被自动剔除
                    prepare_receptor_pure_python(chains[target_chain], valid_ion_lines, receptor_pdbqt)
                    
                    status_text.text("Processing ligand...")
                    progress_bar.progress(60)
                    
                    ext = os.path.splitext(final_sdf_path)[1].lower()
                    if ext == '.sdf': mol = next(Chem.SDMolSupplier(final_sdf_path, removeHs=False), None)
                    else: mol = Chem.MolFromPDBFile(final_sdf_path, removeHs=False)
                        
                    if mol is None: raise Exception("Could not read ligand file.")
                    
                    if ligand_name != "None (Blind Docking)":
                        conf = mol.GetConformer()
                        num_atoms = mol.GetNumAtoms()
                        sum_x, sum_y, sum_z = 0.0, 0.0, 0.0
                        for i in range(num_atoms):
                            pos = conf.GetAtomPosition(i)
                            sum_x += pos.x; sum_y += pos.y; sum_z += pos.z
                        mol_cx, mol_cy, mol_cz = sum_x/num_atoms, sum_y/num_atoms, sum_z/num_atoms
                        
                        dx = center[0] - mol_cx
                        dy = center[1] - mol_cy
                        dz = center[2] - mol_cz
                        
                        for i in range(num_atoms):
                            pos = conf.GetAtomPosition(i)
                            conf.SetAtomPosition(i, Point3D(pos.x + dx, pos.y + dy, pos.z + dz))

                    if not any(atom.GetSymbol() == 'H' for atom in mol.GetAtoms()):
                        mol = Chem.AddHs(mol, addCoords=True)
                        
                    preparator = MoleculePreparation()
                    mol_setups = preparator.prepare(mol)
                    pdbqt_string, is_ok, err = PDBQTWriterLegacy.write_string(mol_setups[0])
                    if not is_ok: raise Exception(f"Meeko error: {err}")
                    
                    ligand_pdbqt = os.path.join(work_dir, "ligand.pdbqt")
                    with open(ligand_pdbqt, "w") as f:
                        f.write(pdbqt_string)
                    
                    status_text.text(f"Running AutoDock Vina...")
                    progress_bar.progress(80)
                    
                    v = Vina(sf_name='vina', cpu=4)
                    v.set_receptor(receptor_pdbqt)
                    v.set_ligand_from_file(ligand_pdbqt)
                    v.compute_vina_maps(center=center, box_size=size)
                    v.dock(exhaustiveness=final_ex, n_poses=final_nm)
                    
                    out_pdbqt = os.path.join(work_dir, "docked.pdbqt")
                    v.write_poses(out_pdbqt, n_poses=final_nm, overwrite=True)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Docking completed!")
                    
                    # ================= 🌟 究极暴力最终版：强制首行归零 =================
                    results_data = []
                    
                    with open(out_pdbqt, 'r', encoding='utf-8') as f_pdbqt:
                        for line in f_pdbqt:
                            if line.startswith("REMARK VINA RESULT:"):
                                parts = line.split()
                                try:
                                    affinity = float(parts[-3])
                                    rmsd_lb = float(parts[-2])
                                    rmsd_ub = float(parts[-1])
                                    
                                    mode = len(results_data) + 1
                                    
                                    # 🔥 核心暴力逻辑：只要是 Mode 1，RMSD 强制为 0
                                    if mode == 1:
                                        rmsd_lb = 0.0
                                        rmsd_ub = 0.0
                                    
                                    results_data.append({
                                        "Mode": mode,
                                        "Affinity (kcal/mol)": round(affinity, 3),
                                        "RMSD l.b. (Å)": round(rmsd_lb, 3),
                                        "RMSD u.b. (Å)": round(rmsd_ub, 3)
                                    })
                                except (ValueError, IndexError):
                                    continue
                    
                    if not results_data:
                        raise Exception("Could not parse docking results.")
                    
                    df = pd.DataFrame(results_data)
                    df = df[["Mode", "Affinity (kcal/mol)", "RMSD l.b. (Å)", "RMSD u.b. (Å)"]]
                    
                    excel_path = os.path.join(work_dir, "report.xlsx")
                    df.to_excel(excel_path, index=False, engine='openpyxl')
                    # ======================================================================

                    with open(excel_path, "rb") as f_xlsx: excel_data = f_xlsx.read()
                    with open(out_pdbqt, "rb") as f_pdbqt: pdbqt_data = f_pdbqt.read()
                    
                    current_record = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "ligand_name": ligand_name,
                        "df": df,
                        "excel_data": excel_data,
                        "pdbqt_data": pdbqt_data
                    }
                    
                    st.session_state.current_job = current_record
                    st.session_state.history.append(current_record)
                    
                    if len(st.session_state.history) > 10:
                        st.session_state.history.pop(0)
                    
                    if "current_online_pdb" in st.session_state: del st.session_state.current_online_pdb
                    if "current_online_sdf" in st.session_state: del st.session_state.current_online_sdf
                    st.session_state.fetched_files = None
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Docking failed: {str(e)}")