BEGIN PLOT /CMS_2026_PAS_SMP_25_010/.*
XLabel=$\log_{10}(\rho^2)$
YLabel=$\frac{1}{\sigma}\frac{\mathrm{d}\sigma}{\mathrm{d}\log_{10}(\rho^2)}$
LogY=0
LegendXPos=0.05
END PLOT

BEGIN PLOT /CMS_2026_PAS_SMP_25_010/zjets_ungroomed_slice0
Title=CMS, 13 TeV, Z + jet, ungroomed, $200 < p_{T}^{jet} < 290$ GeV
END PLOT
BEGIN PLOT /CMS_2026_PAS_SMP_25_010/zjets_ungroomed_slice1
Title=CMS, 13 TeV, Z + jet, ungroomed, $290 < p_{T}^{jet} < 400$ GeV
END PLOT
BEGIN PLOT /CMS_2026_PAS_SMP_25_010/zjets_ungroomed_slice2
Title=CMS, 13 TeV, Z + jet, ungroomed, $p_{T}^{jet} > 400$ GeV
END PLOT
BEGIN PLOT /CMS_2026_PAS_SMP_25_010/zjets_groomed_slice0
Title=CMS, 13 TeV, Z + jet, soft drop, $200 < p_{T}^{jet} < 290$ GeV
END PLOT
BEGIN PLOT /CMS_2026_PAS_SMP_25_010/zjets_groomed_slice1
Title=CMS, 13 TeV, Z + jet, soft drop, $290 < p_{T}^{jet} < 400$ GeV
END PLOT
BEGIN PLOT /CMS_2026_PAS_SMP_25_010/zjets_groomed_slice2
Title=CMS, 13 TeV, Z + jet, soft drop, $p_{T}^{jet} > 400$ GeV
END PLOT

BEGIN PLOT /CMS_2026_PAS_SMP_25_010/zjets_jetpt
Title=CMS, 13 TeV, Z + jet
XLabel=$p_{T}^{jet}$ [GeV]
YLabel=$\mathrm{d}\sigma/\mathrm{d}p_{T}$ [pb/GeV]
LogY=1
END PLOT

BEGIN PLOT /CMS_2026_PAS_SMP_25_010/zjets_zpt
Title=CMS, 13 TeV, Z + jet
XLabel=$p_{T}^{Z}$ [GeV]
YLabel=$\mathrm{d}\sigma/\mathrm{d}p_{T}$ [pb/GeV]
LogY=1
END PLOT

BEGIN PLOT /CMS_2026_PAS_SMP_25_010/zjets_zmass
Title=CMS, 13 TeV, Z + jet
XLabel=$m_{\ell\ell}$ [GeV]
YLabel=$\mathrm{d}\sigma/\mathrm{d}m$ [pb/GeV]
LogY=0
END PLOT
