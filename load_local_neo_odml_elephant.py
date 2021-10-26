import sys

# This loads the Neo and odML libraries shipped with this code. For production
# use, please use the newest releases of odML and Neo.
sys.path.insert(0, 'python-neo')
sys.path.insert(0, 'python-odml')

# This loads the Elephant analysis libraries shipped with this code. It
# is used to generate the offline filtered LFP in example.py.
sys.path.insert(0, 'elephant')
