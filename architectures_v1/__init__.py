from .cells import *
#from .nru import *
from .BRC_Classifier import *
from .GRU_Classifier import *
from .Ladder_Classifier import *
from .Laguerre_Classifier import *
from .LMU_Classifier import *
from .LSTM_Classifier import *
from .nBRC_Classifier import *

from .BRC_SySID import *
from .GRU_SySID import *
from .Ladder_SySID import *
from .Laguerre_SySID import *
from .LMU_SySID import *
from .LSTM_SySID import *
from .nBRC_SySID import *

from .LSTM_Prediction import *
from .LMU_Prediction import *
from .Ladder_Prediction import *
from .GRU_Prediction import *
from .BRC_Prediction import *
from .nBRC_Prediction import *
from .Laguerre_Prediction import *

# List of architectures for classification
LIST_OF_ARCHITECTURES = ["nBRC", "BRC", "LMU", "Laguerre", "Ladder", "LSTM", "GRU"]

# List of architectures for SySID
LIST_OF_ARCHITECTURES_SYSID = ["nBRC", "BRC", "LMU", "Laguerre", "Ladder", "LSTM", "GRU"]

# List of architectures for dynamic prediction
LIST_OF_ARCHITECTURES_DYN_PRED = ["LSTM", "LMU", "Ladder", "GRU", "BRC", "nBRC", "Laguerre"]
