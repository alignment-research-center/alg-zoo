from .architectures import (
    DistMLP,
    DistRNN,
    DistTransformer,
    ScalarMLP,
    ScalarRNN,
    ScalarTransformer,
)
from .handcrafted import handcrafted_2nd_argmax
from .loading import example_2nd_argmax
from .tasks import task_2nd_argmax, task_argmedian, task_longest_cycle, task_median
from .training import train
from .zoo import zoo_2nd_argmax, zoo_argmedian, zoo_longest_cycle, zoo_median
