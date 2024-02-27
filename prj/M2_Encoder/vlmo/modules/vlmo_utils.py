def set_task(pl_module):
    pl_module.current_tasks = [k for k, v in pl_module.hparams.config["loss_names"].items() if v >= 1]
    return


def no_sync_module_apply(module, fn):
    """FSDP module .apply will use _unshard_params_recurse which will sync params across ranks.
    using this function when apply fn is unnecessary to sync params across ranks.
    """
    for child in module.children():
        fn(child)
        no_sync_module_apply(child, fn)
