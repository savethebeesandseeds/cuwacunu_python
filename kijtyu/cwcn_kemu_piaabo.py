# --- --- ---
import os
import sys
import cwcn_config
# --- --- ---
def kemu_assert_dir(__path):
    if not os.path.exists(__path):
        os.makedirs(__path)
    return __path
def kemu_normalize(x):
    x -= x.mean()
    x /= (x.std() + 1e-8)
    return x
def kemu_pretty_print_object(d, indent=0, set_ident=26):
    sys.stdout.write("\n")
    for key, value in d.items():
        sys.stdout.write('\t' * indent + "{}{}{}".format(cwcn_config.CWCN_COLORS.HEADER,str(key),cwcn_config.CWCN_COLORS.REGULAR))
        sys.stdout.write('.' * abs(set_ident-len(str(key))))
        if isinstance(value, dict):
            kemu_pretty_print_object(value, indent+1)
        else:
            sys.stdout.write('\t' * (indent+1) + "{}{}{}\n".format(cwcn_config.CWCN_COLORS.GREEN,str(value),cwcn_config.CWCN_COLORS.REGULAR))
    sys.stdout.flush()