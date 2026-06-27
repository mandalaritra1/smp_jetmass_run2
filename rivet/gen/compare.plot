# Plot directives for the multi-generator overlay (rivet-mkhtml -c).
# Area-normalised shapes; explicit axis labels (rivet-mkhtml isn't given
# RIVET_ANALYSIS_PATH here, so it can't pick them up from the .plot file).

# BEGIN PLOT /CMS_ZJET_JETMASS/mass_u.*
NormalizeToIntegral=1
XLabel=$m_{u}$ [GeV]
YLabel=$1/\sigma\ \mathrm{d}\sigma/\mathrm{d}m$ [1/GeV]
# END PLOT

# BEGIN PLOT /CMS_ZJET_JETMASS/mass_g.*
NormalizeToIntegral=1
XLabel=$m_{g}$ [GeV]
YLabel=$1/\sigma\ \mathrm{d}\sigma/\mathrm{d}m$ [1/GeV]
# END PLOT

# x = 2 log10(m/(pT R))  (rho = m/(pT R) is kept for the ratio itself)
# BEGIN PLOT /CMS_ZJET_JETMASS/rho_u.*
XMin=-4.5
XLabel=$x = 2\log_{10}(m_{u}/(p_{T}R))$
YLabel=$1/\sigma\ \mathrm{d}\sigma/\mathrm{d}x$
# END PLOT

# BEGIN PLOT /CMS_ZJET_JETMASS/rho_g.*
XMin=-4.5
YMax=0.6
XLabel=$x = 2\log_{10}(m_{g}/(p_{T}R))$
YLabel=$1/\sigma\ \mathrm{d}\sigma/\mathrm{d}x$
# END PLOT

# BEGIN PLOT /CMS_ZJET_JETMASS/z_mass
NormalizeToIntegral=1
XLabel=$m_{\ell\ell}$ [GeV]
# END PLOT

# BEGIN PLOT /CMS_ZJET_JETMASS/z_pt
NormalizeToIntegral=1
XLabel=$p_{T}^{Z}$ [GeV]
# END PLOT

# BEGIN PLOT /CMS_ZJET_JETMASS/jet_pt
NormalizeToIntegral=1
XLabel=$p_{T}^{\mathrm{jet}}$ [GeV]
# END PLOT
