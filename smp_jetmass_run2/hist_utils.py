# src/smp_jetmass_run2/hist_utils.py
import hist
import numpy as np


def fill_hist(hdict, name, **kwargs):
    """
    Safely fills a histogram only if it exists in the dictionary.
    """
    if name in hdict:
        hdict[name].fill(**kwargs)


def register_hist(hdict, name, axes, label="Counts"):
    """
    Registers a histogram if not already present in the dictionary.
    """
    from hist import Hist
    if name not in hdict:
        hdict[name] = Hist(*axes, storage="weight", label=label)

def group(h: hist.Hist, oldname: str, newname: str, grouping: dict[str, list[str]]):
    hnew = hist.Hist(
        hist.axis.StrCategory(grouping, name=newname),
        *(ax for ax in h.axes if ax.name != oldname),
        storage=h.storage_type,
    )
    for i, indices in enumerate(grouping.values()):
        hnew.view(flow=True)[i] = h[{oldname: indices}][{oldname: sum}].view(flow=True)

    return hnew

def _refine_edges(edges, factor):
    """Subdivide each bin of ``edges`` into ``factor`` equal sub-bins.

    The original edges are preserved, so the refined binning nests exactly
    inside the original (a refined histogram can be rebinned back to the
    original with no boundary mismatch). factor=1 returns the edges unchanged.
    """
    factor = int(factor)
    edges = np.asarray(edges, dtype=float)
    if factor <= 1:
        return [float(e) for e in edges]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        out.extend(np.linspace(lo, hi, factor, endpoint=False))
    out.append(float(edges[-1]))
    return [float(e) for e in out]


class util_binning :
    '''
    Class to implement the binning schema for jet mass and pt 2d unfolding. The gen-level mass is twice as fine.

    ``rho_refine`` (default 1) subdivides the rho (mpt_gen / mpt_reco) axes by an
    integer factor, preserving the original edges so the fine binning nests in
    the production binning. rho_refine=4 -> 48 gen / 96 reco bins (still
    #reco = 2x #gen). Used by the ``minimal_rho_fine`` mode for the fine-bin /
    fine-then-rebin unfolding study; all other axes are unchanged.
    '''
    def __init__(self, channel="zjet", rho_refine=1):
        self.channel = channel
        self.rho_refine = int(rho_refine)
        self.jetR = 0.8  # AK8 radius, used in rho = 2*log10(m/(pt*R))
        #self.ptreco_axis = hist.axis.Variable([200,260,350,460,550,650,760,13000], name="ptreco", label=r"p_{T,RECO} (GeV)")
        
        #self.mgen_axis = hist.axis.Variable([0, 10, 20, 40, 60, 80, 100, 13000], name="mgen", label=r"Mass (GeV)")

        ### Original version
        #self.mgen_axis = hist.axis.Variable([0, 10, 20, 40, 60, 80, 100, 150, 200, 13000], name="mgen", label=r"Mass (GeV)")
        
        #self.mreco_axis = hist.axis.Variable([0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200, 6200, 13000], name="mreco", label=r"$m_{RECO}$ (GeV)")
        ############

        #self.mgen_axis = hist.axis.Variable([0, 10, 20, 40, 60, 80, 100, 120, 140, 160, 200, 13000], name="mgen", label=r"Mass (GeV)")
        #self.mreco_axis = hist.axis.Variable([0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130 , 140, 150, 160, 180, 200, 250, 300, 350, 400, 500, 1000, 13000], name="mreco", label=r"$m_{RECO}$ (GeV)")

        #self.mgen_axis = hist.axis.Regular(100,0,200, name="mgen", label=r"Mass (GeV)")
        #self.mreco_axis = hist.axis.Regular(100,0,200, name="mreco", label=r"$m_{RECO}$")

        ### Only for making the response matrix for Herwig TOY MC
        #self.mgen_axis = hist.axis.Variable([0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 13000], name="mgen", label=r"Mass (GeV)")

        
        #self.mreco_axis = hist.axis.Variable([0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 13000] , name="mreco", label=r"$m_{RECO}$ (GeV)")
        # Original bins
        # self.mgen_axis = hist.axis.Variable([0, 5, 10, 20, 40, 60, 80, 100, 120, 140,160, 180, 200, 13000], name="mgen", label=r"Mass (GeV)")
        # self.mreco_axis = hist.axis.Variable([0, 2.5, 5, 7.5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 500, 13000] , name="mreco", label=r"$m_{RECO}$ (GeV)")

        self.mgen_axis = hist.axis.Variable([0.0, 10.0, 20.0, 30.0, 50.0, 70.0, 90.0, 110.0, 130.0, 150.0, 170.0, 200.0, 1000.0], name="mgen", label=r"Mass (GeV)")
        self.mreco_axis = hist.axis.Variable([0.0, 5.0, 10.0, 15.0, 20.0, 25.0,  30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 185.0, 200.0, 215.0, 1000.0] , name="mreco", label=r"$m_{RECO}$ (GeV)")

        self.mgen_axis = hist.axis.Variable([0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
                                                12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 
                                                32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 
                                                52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 
                                                72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 
                                                94, 96, 98, 100, 102, 104, 106, 108, 110, 
                                                112, 114, 116, 118, 120, 122, 124, 126, 
                                                128, 130, 132, 134, 136, 138, 140, 142, 
                                                144, 146, 148, 150, 152, 154, 156, 158, 
                                                160, 162, 164, 166, 168, 170, 172, 174, 
                                                176, 178, 180, 182, 184, 186, 188, 190, 
                                                192, 194, 196, 198, 200, 1000], name="mgen", label=r"Mass (GeV)")

        self.mreco_axis = hist.axis.Variable([0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 
                                                    5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 
                                                    10, 11.0, 12, 13.0, 14, 15.0, 16, 17.0, 18, 19.0, 
                                                    20, 21.0, 22, 23.0, 24, 25.0, 26, 27.0, 28, 29.0, 
                                                    30, 31.0, 32, 33.0, 34, 35.0, 36, 37.0, 38, 39.0, 
                                                    40, 41.0, 42, 43.0, 44, 45.0, 46, 47.0, 48, 49.0, 
                                                    50, 51.0, 52, 53.0, 54, 55.0, 56, 57.0, 58, 59.0, 
                                                    60, 61.0, 62, 63.0, 64, 65.0, 66, 67.0, 68, 69.0, 
                                                    70, 71.0, 72, 73.0, 74, 75.0, 76, 77.0, 78.0, 79.0, 80, 81.0, 
                                                    82, 83.0, 84, 85.0, 86, 87.0, 88, 89.0, 90, 91.0, 
                                                    92, 93.0, 94, 95.0, 96, 97.0, 98, 99.0, 100, 101.0, 
                                                    102, 103.0, 104, 105.0, 106, 107.0, 108, 109.0, 110, 111.0, 
                                                    112, 113.0, 114, 115.0, 116, 117.0, 118, 119.0, 120, 121.0, 
                                                    122, 123.0, 124, 125.0, 126, 127.0, 128, 129.0, 130, 131.0, 
                                                    132, 133.0, 134, 135.0, 136, 137.0, 138, 139.0, 140, 141.0, 
                                                    142, 143.0, 144, 145.0, 146, 147.0, 148, 149.0, 150, 151.0, 
                                                    152, 153.0, 154, 155.0, 156, 157.0, 158, 159.0, 160, 161.0, 
                                                    162, 163.0, 164, 165.0, 166, 167.0, 168, 169.0, 170, 171.0, 
                                                    172, 173.0, 174, 175.0, 176, 177.0, 178, 179.0, 180, 181.0, 
                                                    182, 183.0, 184, 185.0, 186, 187.0, 188, 189.0, 190, 191.0, 
                                                    192, 193.0, 194, 195.0, 196, 197.0, 198, 199.0, 200, 500, 1000], name="mreco", label=r"$m_{RECO}$ (GeV)")


        # Negative counterpart of mgen_over_pt_axis (rho_refine subdivides each bin)
        self.mgen_over_pt_axis = hist.axis.Variable(
            _refine_edges(
                [-10, -6, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0],
                self.rho_refine),
            name='mpt_gen', label=r'$\log(\rho^2)$'
        )

        # Groomed truth rho binning.  Its detector-level counterpart below has
        # exactly two bins per truth bin; ungroomed rho retains the axis above.
        # The shown region (-3..-1.5) stays 0.5-wide: finer bins there tripled the
        # unfolded stat error at ~35% purity.  Instead the tail below -3.5 is
        # resolved at 0.25 down to -5 -- a hidden buffer that absorbs low-rho
        # migrations so they do not leak into the last shown bin.  The deep tail
        # (< -5) stays a coarse sink; it is too sparse to bin finely.
        self.mgen_over_pt_g_axis = hist.axis.Variable(
            _refine_edges(
                [-10, -6, -5, -4.75, -4.5, -4.25, -4, -3.75, -3.5,
                 -3, -2.5, -2, -1.5, -1, 0],
                self.rho_refine),
            name='mpt_gen', label=r'$\log(\rho^2)$'
        )

        # Negative counterpart of mreco_over_pt_axis (rho_refine subdivides each bin)
        self.mreco_over_pt_axis = hist.axis.Variable(
            _refine_edges(
                [-10, -8.0, -6, -5.5, -5, -4.75, -4.5, -4.25, -4, -3.75, -3.5, -3.25,
                 -3, -2.75, -2.5, -2.25, -2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0],
                self.rho_refine),
            name='mpt_reco', label=r'$\log(\rho^2)$ (Detector)'
        )

        # Groomed detector rho binning, constructed as an exact 2:1 refinement
        # of mgen_over_pt_g_axis.  This is deliberately distinct from the
        # unchanged ungroomed detector axis above.
        self.mreco_over_pt_g_axis = hist.axis.Variable(
            _refine_edges(
                [-10, -8, -6, -5.5, -5, -4.875, -4.75, -4.625, -4.5, -4.375,
                 -4.25, -4.125, -4, -3.875, -3.75, -3.625, -3.5, -3.25, -3,
                 -2.75, -2.5, -2.25, -2, -1.75, -1.5, -1.25, -1, -0.5, 0],
                self.rho_refine),
            name='mpt_reco', label=r'$\log(\rho^2)$ (Detector)'
        )
 
        
        # ARC round-2: the low pt sink bin is [185,200], not [0,200]. The old
        # [0,200] bin was populated down to the storage floors (FatJet reco floor
        # ~170 GeV, GenJetAK8 gen floor ~100 GeV) -- a wide, data/MC-mismodelled,
        # gen/reco-asymmetric region. Narrowing it to [185,200] keeps the near-200
        # resolution migration inside the response matrix and routes everything
        # further from 200 GeV into the pt underflow, to be handled as a fake
        # (reco<185) or miss (gen<185).
        # NB: this cut creates pt-underflow content for the first time, so the
        # unfold-side fake/miss derivation must sum the matched matrix over the
        # IN-RANGE pt bins only (exclude pt flow); the flow-inclusive projection
        # would otherwise fold cross-boundary events back into "matched" and leak
        # them out of both the matrix and the fakes/misses. See the unfold repo
        # arc_round2 branch (_make_inputs_numpy).
        self.ptgen_axis = hist.axis.Variable([185.,  200.,   290.,   400.,    13000.], name="ptgen", label=r"$p_{T,GEN}$ (GeV)")
        self.ptreco_axis = hist.axis.Variable([185., 200.,   290.,   400.,    13000.], name="ptreco", label=r"$p_{T,RECO}$ (GeV)")

        self.mcut_reco_u_axis = hist.axis.Variable([ 0, 20, 1000], name="mreco", label=r"Mass (GeV)" )
        self.mcut_reco_g_axis = hist.axis.Variable([ 0, 10, 1000], name="mreco", label=r"Mass (GeV)" )

        self.mcut_gen_u_axis = hist.axis.Variable([ 0, 20, 1000], name="mgen", label=r"Mass (GeV)" )
        self.mcut_gen_g_axis = hist.axis.Variable([ 0, 10, 1000], name="mgen", label=r"Mass (GeV)" )

        self.ptgen_axis_fine = hist.axis.Variable([  200., 210,  220, 230, 240, 260., 280, 300, 320, 340,  350., 370, 390, 410, 430, 450, 500, 550, 600, 650, 700, 800, 900, 1000, 13000.], name="ptgen_fine", label=r"$p_{T,GEN}$ (GeV)")  
        
        

        
        self.dataset_axis = hist.axis.StrCategory([], growth=True, name="dataset", label="Primary dataset")
        self.dataset_axis = hist.axis.StrCategory([], growth=True, name="dataset", label="Primary dataset")
        self.lep_axis = hist.axis.StrCategory(["ee", "mm"], name="lep")
        self.n_axis = hist.axis.Regular(10, 0, 10, name="n", label=r"Number")
        self.mass_axis = hist.axis.Regular(100, 0, 500, name="mass", label=r"$m$ [GeV]")
        self.diff_axis = hist.axis.Regular(100, -20, 20, name="diff", label=r"$\Delta$ [GeV]")
        self.diff_axis_large = hist.axis.Regular(100, -50, 50, name="diff", label=r"$\Delta$ [GeV]")
        self.zmass_axis = hist.axis.Regular(40, 70, 110, name="mass", label=r"$m$ [GeV]")
        self.pt_axis = hist.axis.Regular(150, 0, 1500, name="pt", label=r"$p_{T}$ [GeV]")
        self.ptlong_axis = hist.axis.Regular(400, 0, 4000, name="pt", label=r"$H_{T}$ [GeV]")
        self.frac_axis = hist.axis.Regular(200, -3, 3, name="frac", label=r"Fraction")                
        self.mass_ratio_axis = hist.axis.Regular(200, 0, 2, name="mass_ratio", label=r"$m_{groomed}/m_{ungroomed}$")
        self.m_u_reco_5gev_axis = hist.axis.Regular(40, 0, 200, name="m_u_reco", label=r"$m_{ungroomed,RECO}$ [GeV]", underflow=False, overflow=True)
        self.m_g_reco_5gev_axis = hist.axis.Regular(40, 0, 200, name="m_g_reco", label=r"$m_{groomed,RECO}$ [GeV]", underflow=False, overflow=True)
        # Coarse gen groomed/ungroomed mass axes for the groomed<->ungroomed
        # covariance (joint gen-mass histogram, "mass_cov" mode). Same coarse mass
        # binning for both, distinct names so a single 2D (m_g_gen x m_u_gen) hist
        # can carry both; coarse enough to be a fit-ready covariance matrix.
        _m_gen_cov_edges = [0.0, 10.0, 20.0, 30.0, 50.0, 70.0, 90.0, 110.0, 130.0, 150.0, 170.0, 200.0, 1000.0]
        self.m_u_gen_cov_axis = hist.axis.Variable(_m_gen_cov_edges, name="m_u_gen", label=r"$m_{ungroomed,GEN}$ [GeV]")
        self.m_g_gen_cov_axis = hist.axis.Variable(_m_gen_cov_edges, name="m_g_gen", label=r"$m_{groomed,GEN}$ [GeV]")
        # Same idea for rho = 2 log10(m/(pt R)): joint gen groomed/ungroomed rho axes
        # (production rho binning, distinct names) for the rho covariance.
        _rho_gen_cov_edges = [-10, -6, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0]
        self.mpt_u_gen_cov_axis = hist.axis.Variable(_rho_gen_cov_edges, name="mpt_u_gen", label=r"$\log_{10}(\rho^2)_{ungroomed,GEN}$")
        self.mpt_g_gen_cov_axis = hist.axis.Variable(_rho_gen_cov_edges, name="mpt_g_gen", label=r"$\log_{10}(\rho^2)_{groomed,GEN}$")
        # Reco-level counterparts (same coarse binning) for the reco-level
        # groomed<->ungroomed covariance (Scope#3 asks for gen AND reco separately).
        self.m_u_reco_cov_axis = hist.axis.Variable(_m_gen_cov_edges, name="m_u_reco", label=r"$m_{ungroomed,RECO}$ [GeV]")
        self.m_g_reco_cov_axis = hist.axis.Variable(_m_gen_cov_edges, name="m_g_reco", label=r"$m_{groomed,RECO}$ [GeV]")
        self.mpt_u_reco_cov_axis = hist.axis.Variable(_rho_gen_cov_edges, name="mpt_u_reco", label=r"$\log_{10}(\rho^2)_{ungroomed,RECO}$")
        self.mpt_g_reco_cov_axis = hist.axis.Variable(_rho_gen_cov_edges, name="mpt_g_reco", label=r"$\log_{10}(\rho^2)_{groomed,RECO}$")
        self.dr_axis = hist.axis.Regular(150, 0, 6.0, name="dr", label=r"$\Delta R$")
        self.dr_fine_axis = hist.axis.Regular(150, 0, 1.5, name="dr", label=r"$\Delta R$")
        self.dphi_axis = hist.axis.Regular(150, -2*np.pi, 2*np.pi, name="dphi", label=r"$\Delta \phi$")
        self.eta_axis = hist.axis.Regular(40, -2.5, 2.5, name="eta", label=r"$ \eta$")
        self.y_axis = hist.axis.Regular(60, -5, 5, name="y", label="Rapidity $y$")
        self.phi_axis = hist.axis.Regular(40, -5, 5, name="phi", label=r"$ \phi$")
        self.ptfine_axis = hist.axis.Regular(20, 200, 500, name="pt", label=r"p_{T,RECO} [GeV]")
        self.jackknife_axis = hist.axis.IntCategory([], growth = True, name = 'jk', label = "Jackknife categories" )
        
        self.syst_axis=hist.axis.StrCategory([],growth = True, name = "systematic", label = "Systematic Uncertainty")

        # -----------------------------------------------------------------
        # Hadronic (gluon: dijet / trijet) unfolding binning overrides.
        # Single source of truth shared with zjet; default channel="zjet"
        # leaves the axes above untouched.
        #   - pt   : keep the hadronic edges (physics).
        #   - mass : zjet's coarse scheme extended to a high-mass tail
        #            (gluon jets reach higher mass than Z+jet).
        #   - rho  : zjet's DEFINITION + edges (filled as 2*log10(m/(pt*R)))
        #            with extra low-rho bins (-8,-7) for the gluon tail.
        # -----------------------------------------------------------------
        if channel in ("dijet", "trijet"):
            # Low-pt sink is [185,200] as in zjet (ARC round-2 rationale above:
            # keep the near-200 resolution migration inside the response, route
            # everything further below into the pt underflow as fake/miss); the
            # additional high-pt bins beyond zjet's 400 are kept (hadronic
            # triggers reach much higher).  Same unfold-side caveat as zjet:
            # fake/miss-by-subtraction must sum the matched matrix over the
            # IN-RANGE pt bins only (exclude pt flow).
            self.ptgen_axis  = hist.axis.Variable(
                [185., 200., 290., 400., 480., 570., 680., 760., 820., 13000.],
                name="ptgen",  label=r"$p_{T,GEN}$ (GeV)")
            self.ptreco_axis = hist.axis.Variable(
                [185., 200., 290., 400., 480., 570., 680., 760., 820., 13000.],
                name="ptreco", label=r"$p_{T,RECO}$ (GeV)")
            self.mgen_axis  = hist.axis.Variable(
                [0, 10, 20, 30, 50, 70, 90, 110, 130, 150, 170, 200, 300, 500, 13000],
                name="mgen", label=r"Mass (GeV)")
            self.mreco_axis = hist.axis.Variable(
                [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120,
                 130, 140, 150, 160, 170, 185, 200, 215, 300, 400, 500, 13000],
                name="mreco", label=r"$m_{RECO}$ (GeV)")
            self.mgen_over_pt_axis = hist.axis.Variable(
                [-10, -8, -7, -6, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0],
                name="mpt_gen", label=r"$\log(\rho^2)$")
            self.mreco_over_pt_axis = hist.axis.Variable(
                [-10, -8, -7, -6, -5.5, -5, -4.75, -4.5, -4.25, -4, -3.75, -3.5, -3.25,
                 -3, -2.75, -2.5, -2.25, -2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0],
                name="mpt_reco", label=r"$\log(\rho^2)$ (Detector)")
            # Groomed rho: zjet's buffer scheme verbatim (0.5-wide shown region,
            # 0.25-wide hidden buffer resolving -5..-3.5, coarse deep-tail sink)
            # with the hadronic -8/-7 tail edges prepended for the gluon tail.
            # The detector axis is the exact 2:1 refinement of the truth axis,
            # same as zjet (mreco_over_pt_g_axis == gen edges halved).
            _had_rho_gen_g = [-10, -8, -7, -6, -5, -4.75, -4.5, -4.25, -4,
                              -3.75, -3.5, -3, -2.5, -2, -1.5, -1, 0]
            self.mgen_over_pt_g_axis = hist.axis.Variable(
                _refine_edges(_had_rho_gen_g, self.rho_refine),
                name='mpt_gen', label=r'$\log(\rho^2)$')
            self.mreco_over_pt_g_axis = hist.axis.Variable(
                _refine_edges(_refine_edges(_had_rho_gen_g, 2), self.rho_refine),
                name='mpt_reco', label=r'$\log(\rho^2)$ (Detector)')
