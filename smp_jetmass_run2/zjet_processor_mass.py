from .zjet_processor import QJetMassProcessor as _QJetMassProcessor


class QJetMassProcessor(_QJetMassProcessor):
    """
    Mass-only entry point.

    This keeps the event selection and correction logic shared with the main
    processor, but constrains the public modes to the mass histogram profiles so
    we do not accidentally allocate rho histograms in a mass-only run.
    """

    _MODE_ALIASES = {
        "mass": "minimal",
        "mass_reweight": "reweight_pythia",
        "mass_jk": "mass_jk",
        "mass_jk_mc": "mass_jk",
        "mass_jk_data": "mass_jk",
        "mass_diagnostic_ntuple": "mass_diagnostic_ntuple",
    }

    _ALLOWED_MODES = {
        "minimal",
        "reweight_pythia",
        "mass_jk",
        "jk_mc",
        "jk_data",
        "mass_diagnostic_ntuple",
    }

    def __init__(
        self,
        do_gen=True,
        mode="mass",
        debug=False,
        jet_systematics=None,
        systematics=None,
    ):
        resolved_mode = self._MODE_ALIASES.get(mode, mode)
        if resolved_mode not in self._ALLOWED_MODES:
            allowed = ", ".join(
                sorted(self._ALLOWED_MODES | set(self._MODE_ALIASES))
            )
            raise ValueError(
                f"Mass processor only supports mass modes. Got '{mode}'. "
                f"Allowed values: {allowed}"
            )

        super().__init__(
            do_gen=do_gen,
            mode=resolved_mode,
            debug=debug,
            jet_systematics=jet_systematics,
            systematics=systematics,
        )
