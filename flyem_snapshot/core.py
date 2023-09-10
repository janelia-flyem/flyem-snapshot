ConfigSchema = {
    "description": "Configuration for exporting connectomic denormalizations from DVID",
    "default": {},
    "required": ["snapshot", "synapse-points", "synapse-partners"],
    "additionalProperties": False,
    "properties": {
        "snapshot-tag": {
            "description":
                "A suffix to add to export filenames.\n"
                "By default, a tag is automatically chosen which incorporates the\n"
                "snapshot date, uuid, and uuid commit status.",
            "type": "string",
            "default": "",
        },
        "output-dir": {
            "description":
                "Where to write output feather files.\n"
                "Relative paths here are interpreted from the directory in which this config file is stored.\n"
                "If not specified, a reasonable default is chosen IN THE SAME DIRECTORY AS THIS CONFIG FILE.\n",
            "type": "string",
            "default": "",
        },
        "neuprint": NeuprintSchema,
        "flat-files": FlatFileConfigSchema,
        "reports": ReportsSchema,
        "body-size-cache": {
            "type": "object",
            "description":
                "An on-disk cache for body sizes, obtained from a prior uuid.\n"
                "We'll have to fetch sizes only for those bodies which have been modified since this cache was created.\n",
            "default": {},
            "additionalProperties": False,
            "properties": {
                "file": {
                    "description": "Feather file with columns 'body' and 'size'",
                    "type": "string",
                    "default": "",
                },
                "uuid": {
                    "description": "The locked uuid from which these sizes were obtained.\n",
                    "type": "string",
                    "default": "",
                },
            }
        },
        "processes": {
            "description":
                "For steps which benefit from multiprocessing, how many processes should be used?",
            "type": "integer",
            "default": 16,
        },
        "dvid-timeout": {
            "description": "Timeout for dvid requests, in seconds. Used for both 'connect' and 'read' timeout.",
            "type": "number",
            "default": 180.0,
        },
    }
}


