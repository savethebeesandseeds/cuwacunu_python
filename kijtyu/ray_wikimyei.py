# --- ---- --- 
import os
import sys
# --- ---- --- 
# CWCN_KIJTIYU_FOLDER="../kijtyu"
# os.environ['CWCN_KIJTYU_FOLDER']=os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)),CWCN_KIJTIYU_FOLDER))
# sys.path.append(os.environ['CWCN_KIJTYU_FOLDER'])
# --- ---- --- 
import cwcn_config
import cwcn_wikimyei_nebajke
import cwcn_wikimyei_piaabo
import cwcn_jkimyei_nebajke
import cwcn_tsinuu_piaabo
import cwcn_kemu_piaabo
# --- ---- --- 
os.environ['CWCN_CONFIG_SYSTEM']='ray_system' # redundant
# --- ---- --- 
if __name__ == "__main__":
    cwcn_kemu_piaabo.kemu_assert_dir(cwcn_config.CWCN_CONFIG().CHECKPOINTS_FOLDER)
    c_ray_jkimyei=cwcn_jkimyei_nebajke.RAY_ORDER_JKIMYEI()
    # c_ray_jkimyei._ray_main_()
    c_ray_jkimyei._read_ray_logs_()
    