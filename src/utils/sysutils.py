import multiprocessing
import sys

def is_debug_mode():
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is None:
        return False
    elif gettrace():
        # print('Hmm, Big Debugger is watching me')
        return True
    else:
        return False

def get_cores_count():
    # PyCharm debugging doesn't work with Pytorch when looping dataset using dataloader with workers set to non-zero
    return 0 if is_debug_mode() else multiprocessing.cpu_count()

print(f"Cores count = {get_cores_count()}")