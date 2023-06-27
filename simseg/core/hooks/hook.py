class Hook(object):
    r"""
    Basic Hook object.
    """

    def init_runner(self, runner):
        pass

    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass

    def before_epoch(self, runner, epoch_state):
        pass

    def after_epoch(self, runner, epoch_state):
        pass

    def before_step(self, runner, epoch_state, step_state):
        pass

    def after_step(self, runner, epoch_state, step_state):
        pass

    def before_train_epoch(self, runner, epoch_state):
        pass

    def before_val_epoch(self, runner, epoch_state):
        pass

    def after_train_epoch(self, runner, epoch_state):
        pass

    def after_val_epoch(self, runner, epoch_state):
        pass

    def before_train_step(self, runner, epoch_state, step_state):
        pass

    def before_val_step(self, runner, epoch_state, step_state):
        pass

    def after_train_step(self, runner, epoch_state, step_state):
        pass

    def after_val_step(self, runner, epoch_state, step_state):
        pass

    def _before_train_epoch(self, runner, epoch_state):
        self.before_epoch(runner, epoch_state)
        self.before_train_epoch(runner, epoch_state)

    def _before_val_epoch(self, runner, epoch_state):
        self.before_epoch(runner, epoch_state)
        self.before_val_epoch(runner, epoch_state)

    def _after_train_epoch(self, runner, epoch_state):
        self.after_epoch(runner, epoch_state)
        self.after_train_epoch(runner, epoch_state)

    def _after_val_epoch(self, runner, epoch_state):
        self.after_epoch(runner, epoch_state)
        self.after_val_epoch(runner, epoch_state)

    def _before_train_step(self, runner, epoch_state, step_state):
        self.before_step(runner, epoch_state, step_state)
        self.before_train_step(runner, epoch_state, step_state)

    def _before_val_step(self, runner, epoch_state, step_state):
        self.before_step(runner, epoch_state, step_state)
        self.before_val_step(runner, epoch_state, step_state)

    def _after_train_step(self, runner, epoch_state, step_state):
        self.after_step(runner, epoch_state, step_state)
        self.after_train_step(runner, epoch_state, step_state)

    def _after_val_step(self, runner, epoch_state, step_state):
        self.after_step(runner, epoch_state, step_state)
        self.after_val_step(runner, epoch_state, step_state)

    @staticmethod
    def every_n_epochs(runner, n):
        return n > 0 and (runner.epoch+1) % n == 0

    @staticmethod
    def every_n_inner_steps(epoch_state, n):
        return n > 0 and (epoch_state.inner_step+1) % n == 0

    @staticmethod
    def every_n_steps(runner, n):
        return n > 0 and (runner.step+1) % n == 0
