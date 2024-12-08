from . import io
from . import preprocessing
from . import markers
from . import reductions
from . import report
from . import predict

io.register()
preprocessing.register()
markers.register()
reductions.register()
report.register()
predict.register()
