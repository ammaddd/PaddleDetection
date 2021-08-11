import logging
import functools
try:
    import comet_ml
    comet_installed = True
except:
    comet_installed = False


class CometLogger:
    def __init__(self, comet=False, **kwargs):
        global comet_installed
        self._logging = None
        self._comet_args = kwargs
        if comet == False:
            self._logging = False
        elif comet == True and comet_installed == False:
            raise Exception("Comet not installed. Run 'pip install comet-ml'")
    
    def _requiresComet(method):
        @functools.wraps(method)
        def wrapper(*args, **kwargs):
            self = args[0]
            global comet_installed
            if self._logging is None and comet_installed:
                self._logging = False
                try:
                    comet_ml.init()
                    if comet_ml.get_global_experiment() is not None:
                        logging.warning("You have already created a comet \
                                        experiment manually, which might \
                                        cause clashes")
                    self._experiment = comet_ml.Experiment(**self._comet_args)
                    self._logging = True
                except Exception as e:
                    logging.warning(e)

            if self._logging == True:
                return method(*args, **kwargs)
        return wrapper
    
    @_requiresComet
    def end(self):
        """Ends an experiment."""
        self._experiment.end()
        comet_ml.config.experiment = None

    @_requiresComet
    def log_others(self, dictionary):
        """Reports dictionary of key/values to the Other tab on Comet.ml.
        Useful for reporting datasets attributes, datasets path, unique identifiers etc.

        Args:
            dictionary:  dict of key/values where value is Any type
              of value (str,int,float..)
        """
        self._experiment.log_others(dictionary)

    @_requiresComet
    def log_metric(self, name, value, step=None, epoch=None,
                   include_context=True):
        """Logs a general metric (i.e accuracy, f1)..

        Args:
            name: String - Name of your metric
            value: Float/Integer/Boolean/String
            step: Optional. Used as the X axis when plotting on comet.ml
            epoch: Optional. Used as the X axis when plotting on comet.ml
            include_context: Optional. If set to True (the default),
                the current context will be logged along the metric.
        """
        self._experiment.log_metric(name, value, step, epoch,
                                    include_context)

    @_requiresComet
    def log_model(self, name, file_or_folder, file_name=None):
        """Logs the model data under the name. Data can be a file path,
            a folder path or a file-like object.

        Args:         
            name: string (required), the name of the model
            file_or_folder: the model data (required); can be a file path,
                a folder path or a file-like object.
            file_name: (optional) the name of the model data. Used with.
        """
        self._experiment.log_model(name, file_or_folder, file_name)

    @_requiresComet
    def log_asset_data(self, data, name=None, overwrite=False, step=None,
                       metadata=None, epoch=None):
        """Logs the data given (str, binary, or JSON).
        Args:  
        data: data to be saved as asset
        name: String, optional. A custom file name to be displayed If
            not provided the filename from the temporary saved file
            will be used.
        overwrite: Boolean, optional. Default False. If True will
            overwrite all existing assets with the same name.
        step: Optional. Used to associate the asset to a specific step.
        epoch: Optional. Used to associate the asset to a specific epoch.
        metadata: Optional. Some additional data to attach to the
            asset data. Must be a JSON-encodable dict.
        """
        self._experiment.log_asset_data(data, name, overwrite, step,
                                        metadata, epoch)

    @_requiresComet
    def log_metrics(self, dic, prefix=None, step=None, epoch=None):
        """Logs a key,value dictionary of metrics.
        Args:
            dic: key,value dictionary of metrics
            prefix: prefix added to metric name
            step: Optional. Used as the X axis when plotting on comet.ml
            epoch: Optional. Used as the X axis when plotting on comet.ml
        """
        self._experiment.log_metrics(dic, prefix, step, epoch)
