"""
Script created by PeterC to test the torch profiler 08-11-2024. 
Reference: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
"""
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity


def profile_example_cpu():
    model = models.resnet18()
    inputs = torch.randn(5, 3, 224, 224)

    # Define profiling scope and setup profiler
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            model(inputs)

    # Print a summary of the profiling
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

def profile_example_exetime():

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.xpu.is_available():
        device = 'xpu'
    else:
        print('Neither CUDA nor XPU devices are available to demonstrate profiling on acceleration devices')
        import sys
        sys.exit(0)

    activities = [ProfilerActivity.CPU,
                ProfilerActivity.CUDA, ProfilerActivity.XPU]
    sort_by_keyword = device + "_time_total"

    model = models.resnet18().to(device)
    inputs = torch.randn(5, 3, 224, 224).to(device)

    with profile(activities=activities, record_shapes=True) as prof:
        with record_function("model_inference"):
            model(inputs)

    print(prof.key_averages().table(sort_by=sort_by_keyword, row_limit=10))

def profile_example_memory():
    model = models.resnet18()
    inputs = torch.randn(5, 3, 224, 224)

    with profile(activities=[ProfilerActivity.CPU],
                profile_memory=True, record_shapes=True) as prof:
        model(inputs)

    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

    # (omitting some columns)
    # ---------------------------------  ------------  ------------  ------------
    #                              Name       CPU Mem  Self CPU Mem    # of Calls
    # ---------------------------------  ------------  ------------  ------------
    #                       aten::empty      94.79 Mb      94.79 Mb           121
    #     aten::max_pool2d_with_indices      11.48 Mb      11.48 Mb             1
    #                       aten::addmm      19.53 Kb      19.53 Kb             1
    #               aten::empty_strided         572 b         572 b            25
    #                     aten::resize_         240 b         240 b             6
    #                         aten::abs         480 b         240 b             4
    #                         aten::add         160 b         160 b            20
    #               aten::masked_select         120 b         112 b             1
    #                          aten::ne         122 b          53 b             6
    #                          aten::eq          60 b          30 b             2
    # ---------------------------------  ------------  ------------  ------------
    # Self CPU time total: 53.064ms

    print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))


def profile_example_tracingProf():
    device = 'cuda'
    activities = [ProfilerActivity.CPU,
                ProfilerActivity.CUDA, ProfilerActivity.XPU]

    model = models.resnet18().to(device)
    inputs = torch.randn(5, 3, 224, 224).to(device)

    with profile(activities=activities) as prof:
        model(inputs)

    prof.export_chrome_trace("./trace.json")

def profile_example_stackTrace():

    device = 'cuda'
    activities = [ProfilerActivity.CPU,
                ProfilerActivity.CUDA, ProfilerActivity.XPU]

    model = models.resnet18().to(device)
    inputs = torch.randn(5, 3, 224, 224).to(device)

    sort_by_keyword = "self_" + device + "_time_total"
    with profile(
        activities=activities,
        with_stack=True,
    ) as prof:
        model(inputs)

    # Print aggregated stats
    print(prof.key_averages(group_by_stack_n=5).table(
        sort_by=sort_by_keyword, row_limit=2))

def profile_example_longJobs():
    from torch.profiler import schedule
    my_schedule = schedule(
        skip_first=10,
        wait=5,
        warmup=1,
        active=3,
        repeat=2)

    sort_by_keyword = "self_" + device + "_time_total"
    
    def trace_handler(p):
        output = p.key_averages().table(sort_by=sort_by_keyword, row_limit=10)
        print(output)
        p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")


        with profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=2),
            on_trace_ready=trace_handler
        ) as p:
            for idx in range(8):
                model(inputs)
                p.step()


if __name__ == '__main__':
    profile_example_cpu()