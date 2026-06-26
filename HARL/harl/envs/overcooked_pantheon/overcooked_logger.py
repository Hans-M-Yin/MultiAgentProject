from harl.common.base_logger import BaseLogger


class OvercookedPantheonLogger(BaseLogger):
    def get_task_name(self):
        return self.env_args["layout_name"]
