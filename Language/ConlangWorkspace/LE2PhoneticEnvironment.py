class PhoneticEnvironment:
    def __init__(self, before_environment, after_environment):
        self.before_environment = before_environment
        self.after_environment = after_environment
        # should be able to match underspecified feature dicts
