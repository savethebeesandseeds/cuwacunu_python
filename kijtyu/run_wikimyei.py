# --- ---- --- 
# test_wikimyei
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
os.environ['CWCN_CONFIG_SYSTEM']='default' # change to RAY or set to default
# --- ---- --- 
if __name__ == "__main__":
    # --- --- --- --- --- 
    cwcn_kemu_piaabo.kemu_assert_dir(cwcn_config.CWCN_CONFIG().CHECKPOINTS_FOLDER)
    c_wikimyei=cwcn_wikimyei_nebajke.WIKIMYEI(cwcn_config.CWCN_CONFIG().__dict__,_load_file=cwcn_config.c_load_file)
    c_wikimyei.jkimyei._standalone_wikimyei_jkimyei_ppo_loop_()
    # --- --- --- --- --- 
    repeat =None
    while(repeat and repeat is not None):
        try:
            c_wikimyei._standalone_wikimyei_loop_()
            repeat=False
        except Exception as e:
            print(e)
            repeat=True
    