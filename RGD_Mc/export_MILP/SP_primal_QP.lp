\ Model SP_primal_QP_gurobi
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  x_cost_ESS[0] + x_cost_ESS[1] + x_cost_ESS[2] + x_cost_ESS[3]
   + x_cost_ESS[4] + x_cost_ESS[5] + x_cost_ESS[6] + x_cost_ESS[7]
   + x_cost_ESS[8] + x_cost_ESS[9] + x_cost_ESS[10] + x_cost_ESS[11]
   + x_cost_ESS[12] + x_cost_ESS[13] + x_cost_ESS[14] + x_cost_ESS[15]
   + x_cost_ESS[16] + x_cost_ESS[17] + x_cost_ESS[18] + x_cost_ESS[19]
   + x_cost_ESS[20] + x_cost_ESS[21] + x_cost_ESS[22] + x_cost_ESS[23]
   + x_cost_ESS[24] + x_cost_ESS[25] + x_cost_ESS[26] + x_cost_ESS[27]
   + x_cost_ESS[28] + x_cost_ESS[29] + x_cost_ESS[30] + x_cost_ESS[31]
   + x_cost_ESS[32] + x_cost_ESS[33] + x_cost_ESS[34] + x_cost_ESS[35]
   + x_cost_ESS[36] + x_cost_ESS[37] + x_cost_ESS[38] + x_cost_ESS[39]
   + x_cost_ESS[40] + x_cost_ESS[41] + x_cost_ESS[42] + x_cost_ESS[43]
   + x_cost_ESS[44] + x_cost_ESS[45] + x_cost_ESS[46] + x_cost_ESS[47]
   + x_cost_ESS[48] + x_cost_ESS[49] + x_cost_ESS[50] + x_cost_ESS[51]
   + x_cost_ESS[52] + x_cost_ESS[53] + x_cost_ESS[54] + x_cost_ESS[55]
   + x_cost_ESS[56] + x_cost_ESS[57] + x_cost_ESS[58] + x_cost_ESS[59]
   + x_cost_ESS[60] + x_cost_ESS[61] + x_cost_ESS[62] + x_cost_ESS[63]
   + x_cost_ESS[64] + x_cost_ESS[65] + x_cost_ESS[66] + x_cost_ESS[67]
   + x_cost_ESS[68] + x_cost_ESS[69] + x_cost_ESS[70] + x_cost_ESS[71]
   + x_cost_ESS[72] + x_cost_ESS[73] + x_cost_ESS[74] + x_cost_ESS[75]
   + x_cost_ESS[76] + x_cost_ESS[77] + x_cost_ESS[78] + x_cost_ESS[79]
   + x_cost_ESS[80] + x_cost_ESS[81] + x_cost_ESS[82] + x_cost_ESS[83]
   + x_cost_ESS[84] + x_cost_ESS[85] + x_cost_ESS[86] + x_cost_ESS[87]
   + x_cost_ESS[88] + x_cost_ESS[89] + x_cost_ESS[90] + x_cost_ESS[91]
   + x_cost_ESS[92] + x_cost_ESS[93] + x_cost_ESS[94] + x_cost_ESS[95]
   + x_cost_fuel_PWL[0] + x_cost_fuel_PWL[1] + x_cost_fuel_PWL[2]
   + x_cost_fuel_PWL[3] + x_cost_fuel_PWL[4] + x_cost_fuel_PWL[5]
   + x_cost_fuel_PWL[6] + x_cost_fuel_PWL[7] + x_cost_fuel_PWL[8]
   + x_cost_fuel_PWL[9] + x_cost_fuel_PWL[10] + x_cost_fuel_PWL[11]
   + x_cost_fuel_PWL[12] + x_cost_fuel_PWL[13] + x_cost_fuel_PWL[14]
   + x_cost_fuel_PWL[15] + x_cost_fuel_PWL[16] + x_cost_fuel_PWL[17]
   + x_cost_fuel_PWL[18] + x_cost_fuel_PWL[19] + x_cost_fuel_PWL[20]
   + x_cost_fuel_PWL[21] + x_cost_fuel_PWL[22] + x_cost_fuel_PWL[23]
   + x_cost_fuel_PWL[24] + x_cost_fuel_PWL[25] + x_cost_fuel_PWL[26]
   + x_cost_fuel_PWL[27] + x_cost_fuel_PWL[28] + x_cost_fuel_PWL[29]
   + x_cost_fuel_PWL[30] + x_cost_fuel_PWL[31] + x_cost_fuel_PWL[32]
   + x_cost_fuel_PWL[33] + x_cost_fuel_PWL[34] + x_cost_fuel_PWL[35]
   + x_cost_fuel_PWL[36] + x_cost_fuel_PWL[37] + x_cost_fuel_PWL[38]
   + x_cost_fuel_PWL[39] + x_cost_fuel_PWL[40] + x_cost_fuel_PWL[41]
   + x_cost_fuel_PWL[42] + x_cost_fuel_PWL[43] + x_cost_fuel_PWL[44]
   + x_cost_fuel_PWL[45] + x_cost_fuel_PWL[46] + x_cost_fuel_PWL[47]
   + x_cost_fuel_PWL[48] + x_cost_fuel_PWL[49] + x_cost_fuel_PWL[50]
   + x_cost_fuel_PWL[51] + x_cost_fuel_PWL[52] + x_cost_fuel_PWL[53]
   + x_cost_fuel_PWL[54] + x_cost_fuel_PWL[55] + x_cost_fuel_PWL[56]
   + x_cost_fuel_PWL[57] + x_cost_fuel_PWL[58] + x_cost_fuel_PWL[59]
   + x_cost_fuel_PWL[60] + x_cost_fuel_PWL[61] + x_cost_fuel_PWL[62]
   + x_cost_fuel_PWL[63] + x_cost_fuel_PWL[64] + x_cost_fuel_PWL[65]
   + x_cost_fuel_PWL[66] + x_cost_fuel_PWL[67] + x_cost_fuel_PWL[68]
   + x_cost_fuel_PWL[69] + x_cost_fuel_PWL[70] + x_cost_fuel_PWL[71]
   + x_cost_fuel_PWL[72] + x_cost_fuel_PWL[73] + x_cost_fuel_PWL[74]
   + x_cost_fuel_PWL[75] + x_cost_fuel_PWL[76] + x_cost_fuel_PWL[77]
   + x_cost_fuel_PWL[78] + x_cost_fuel_PWL[79] + x_cost_fuel_PWL[80]
   + x_cost_fuel_PWL[81] + x_cost_fuel_PWL[82] + x_cost_fuel_PWL[83]
   + x_cost_fuel_PWL[84] + x_cost_fuel_PWL[85] + x_cost_fuel_PWL[86]
   + x_cost_fuel_PWL[87] + x_cost_fuel_PWL[88] + x_cost_fuel_PWL[89]
   + x_cost_fuel_PWL[90] + x_cost_fuel_PWL[91] + x_cost_fuel_PWL[92]
   + x_cost_fuel_PWL[93] + x_cost_fuel_PWL[94] + x_cost_fuel_PWL[95]
   + x_cost_res[0] + x_cost_res[1] + x_cost_res[2] + x_cost_res[3]
   + x_cost_res[4] + x_cost_res[5] + x_cost_res[6] + x_cost_res[7]
   + x_cost_res[8] + x_cost_res[9] + x_cost_res[10] + x_cost_res[11]
   + x_cost_res[12] + x_cost_res[13] + x_cost_res[14] + x_cost_res[15]
   + x_cost_res[16] + x_cost_res[17] + x_cost_res[18] + x_cost_res[19]
   + x_cost_res[20] + x_cost_res[21] + x_cost_res[22] + x_cost_res[23]
   + x_cost_res[24] + x_cost_res[25] + x_cost_res[26] + x_cost_res[27]
   + x_cost_res[28] + x_cost_res[29] + x_cost_res[30] + x_cost_res[31]
   + x_cost_res[32] + x_cost_res[33] + x_cost_res[34] + x_cost_res[35]
   + x_cost_res[36] + x_cost_res[37] + x_cost_res[38] + x_cost_res[39]
   + x_cost_res[40] + x_cost_res[41] + x_cost_res[42] + x_cost_res[43]
   + x_cost_res[44] + x_cost_res[45] + x_cost_res[46] + x_cost_res[47]
   + x_cost_res[48] + x_cost_res[49] + x_cost_res[50] + x_cost_res[51]
   + x_cost_res[52] + x_cost_res[53] + x_cost_res[54] + x_cost_res[55]
   + x_cost_res[56] + x_cost_res[57] + x_cost_res[58] + x_cost_res[59]
   + x_cost_res[60] + x_cost_res[61] + x_cost_res[62] + x_cost_res[63]
   + x_cost_res[64] + x_cost_res[65] + x_cost_res[66] + x_cost_res[67]
   + x_cost_res[68] + x_cost_res[69] + x_cost_res[70] + x_cost_res[71]
   + x_cost_res[72] + x_cost_res[73] + x_cost_res[74] + x_cost_res[75]
   + x_cost_res[76] + x_cost_res[77] + x_cost_res[78] + x_cost_res[79]
   + x_cost_res[80] + x_cost_res[81] + x_cost_res[82] + x_cost_res[83]
   + x_cost_res[84] + x_cost_res[85] + x_cost_res[86] + x_cost_res[87]
   + x_cost_res[88] + x_cost_res[89] + x_cost_res[90] + x_cost_res[91]
   + x_cost_res[92] + x_cost_res[93] + x_cost_res[94] + x_cost_res[95]
   + x_cost_cut_PWL[0] + x_cost_cut_PWL[1] + x_cost_cut_PWL[2]
   + x_cost_cut_PWL[3] + x_cost_cut_PWL[4] + x_cost_cut_PWL[5]
   + x_cost_cut_PWL[6] + x_cost_cut_PWL[7] + x_cost_cut_PWL[8]
   + x_cost_cut_PWL[9] + x_cost_cut_PWL[10] + x_cost_cut_PWL[11]
   + x_cost_cut_PWL[12] + x_cost_cut_PWL[13] + x_cost_cut_PWL[14]
   + x_cost_cut_PWL[15] + x_cost_cut_PWL[16] + x_cost_cut_PWL[17]
   + x_cost_cut_PWL[18] + x_cost_cut_PWL[19] + x_cost_cut_PWL[20]
   + x_cost_cut_PWL[21] + x_cost_cut_PWL[22] + x_cost_cut_PWL[23]
   + x_cost_cut_PWL[24] + x_cost_cut_PWL[25] + x_cost_cut_PWL[26]
   + x_cost_cut_PWL[27] + x_cost_cut_PWL[28] + x_cost_cut_PWL[29]
   + x_cost_cut_PWL[30] + x_cost_cut_PWL[31] + x_cost_cut_PWL[32]
   + x_cost_cut_PWL[33] + x_cost_cut_PWL[34] + x_cost_cut_PWL[35]
   + x_cost_cut_PWL[36] + x_cost_cut_PWL[37] + x_cost_cut_PWL[38]
   + x_cost_cut_PWL[39] + x_cost_cut_PWL[40] + x_cost_cut_PWL[41]
   + x_cost_cut_PWL[42] + x_cost_cut_PWL[43] + x_cost_cut_PWL[44]
   + x_cost_cut_PWL[45] + x_cost_cut_PWL[46] + x_cost_cut_PWL[47]
   + x_cost_cut_PWL[48] + x_cost_cut_PWL[49] + x_cost_cut_PWL[50]
   + x_cost_cut_PWL[51] + x_cost_cut_PWL[52] + x_cost_cut_PWL[53]
   + x_cost_cut_PWL[54] + x_cost_cut_PWL[55] + x_cost_cut_PWL[56]
   + x_cost_cut_PWL[57] + x_cost_cut_PWL[58] + x_cost_cut_PWL[59]
   + x_cost_cut_PWL[60] + x_cost_cut_PWL[61] + x_cost_cut_PWL[62]
   + x_cost_cut_PWL[63] + x_cost_cut_PWL[64] + x_cost_cut_PWL[65]
   + x_cost_cut_PWL[66] + x_cost_cut_PWL[67] + x_cost_cut_PWL[68]
   + x_cost_cut_PWL[69] + x_cost_cut_PWL[70] + x_cost_cut_PWL[71]
   + x_cost_cut_PWL[72] + x_cost_cut_PWL[73] + x_cost_cut_PWL[74]
   + x_cost_cut_PWL[75] + x_cost_cut_PWL[76] + x_cost_cut_PWL[77]
   + x_cost_cut_PWL[78] + x_cost_cut_PWL[79] + x_cost_cut_PWL[80]
   + x_cost_cut_PWL[81] + x_cost_cut_PWL[82] + x_cost_cut_PWL[83]
   + x_cost_cut_PWL[84] + x_cost_cut_PWL[85] + x_cost_cut_PWL[86]
   + x_cost_cut_PWL[87] + x_cost_cut_PWL[88] + x_cost_cut_PWL[89]
   + x_cost_cut_PWL[90] + x_cost_cut_PWL[91] + x_cost_cut_PWL[92]
   + x_cost_cut_PWL[93] + x_cost_cut_PWL[94] + x_cost_cut_PWL[95]
   + y_cost_fuel[0] + y_cost_fuel[1] + y_cost_fuel[2] + y_cost_fuel[3]
   + y_cost_fuel[4] + y_cost_fuel[5] + y_cost_fuel[6] + y_cost_fuel[7]
   + y_cost_fuel[8] + y_cost_fuel[9] + y_cost_fuel[10] + y_cost_fuel[11]
   + y_cost_fuel[12] + y_cost_fuel[13] + y_cost_fuel[14] + y_cost_fuel[15]
   + y_cost_fuel[16] + y_cost_fuel[17] + y_cost_fuel[18] + y_cost_fuel[19]
   + y_cost_fuel[20] + y_cost_fuel[21] + y_cost_fuel[22] + y_cost_fuel[23]
   + y_cost_fuel[24] + y_cost_fuel[25] + y_cost_fuel[26] + y_cost_fuel[27]
   + y_cost_fuel[28] + y_cost_fuel[29] + y_cost_fuel[30] + y_cost_fuel[31]
   + y_cost_fuel[32] + y_cost_fuel[33] + y_cost_fuel[34] + y_cost_fuel[35]
   + y_cost_fuel[36] + y_cost_fuel[37] + y_cost_fuel[38] + y_cost_fuel[39]
   + y_cost_fuel[40] + y_cost_fuel[41] + y_cost_fuel[42] + y_cost_fuel[43]
   + y_cost_fuel[44] + y_cost_fuel[45] + y_cost_fuel[46] + y_cost_fuel[47]
   + y_cost_fuel[48] + y_cost_fuel[49] + y_cost_fuel[50] + y_cost_fuel[51]
   + y_cost_fuel[52] + y_cost_fuel[53] + y_cost_fuel[54] + y_cost_fuel[55]
   + y_cost_fuel[56] + y_cost_fuel[57] + y_cost_fuel[58] + y_cost_fuel[59]
   + y_cost_fuel[60] + y_cost_fuel[61] + y_cost_fuel[62] + y_cost_fuel[63]
   + y_cost_fuel[64] + y_cost_fuel[65] + y_cost_fuel[66] + y_cost_fuel[67]
   + y_cost_fuel[68] + y_cost_fuel[69] + y_cost_fuel[70] + y_cost_fuel[71]
   + y_cost_fuel[72] + y_cost_fuel[73] + y_cost_fuel[74] + y_cost_fuel[75]
   + y_cost_fuel[76] + y_cost_fuel[77] + y_cost_fuel[78] + y_cost_fuel[79]
   + y_cost_fuel[80] + y_cost_fuel[81] + y_cost_fuel[82] + y_cost_fuel[83]
   + y_cost_fuel[84] + y_cost_fuel[85] + y_cost_fuel[86] + y_cost_fuel[87]
   + y_cost_fuel[88] + y_cost_fuel[89] + y_cost_fuel[90] + y_cost_fuel[91]
   + y_cost_fuel[92] + y_cost_fuel[93] + y_cost_fuel[94] + y_cost_fuel[95]
   + y_cost_ESS[0] + y_cost_ESS[1] + y_cost_ESS[2] + y_cost_ESS[3]
   + y_cost_ESS[4] + y_cost_ESS[5] + y_cost_ESS[6] + y_cost_ESS[7]
   + y_cost_ESS[8] + y_cost_ESS[9] + y_cost_ESS[10] + y_cost_ESS[11]
   + y_cost_ESS[12] + y_cost_ESS[13] + y_cost_ESS[14] + y_cost_ESS[15]
   + y_cost_ESS[16] + y_cost_ESS[17] + y_cost_ESS[18] + y_cost_ESS[19]
   + y_cost_ESS[20] + y_cost_ESS[21] + y_cost_ESS[22] + y_cost_ESS[23]
   + y_cost_ESS[24] + y_cost_ESS[25] + y_cost_ESS[26] + y_cost_ESS[27]
   + y_cost_ESS[28] + y_cost_ESS[29] + y_cost_ESS[30] + y_cost_ESS[31]
   + y_cost_ESS[32] + y_cost_ESS[33] + y_cost_ESS[34] + y_cost_ESS[35]
   + y_cost_ESS[36] + y_cost_ESS[37] + y_cost_ESS[38] + y_cost_ESS[39]
   + y_cost_ESS[40] + y_cost_ESS[41] + y_cost_ESS[42] + y_cost_ESS[43]
   + y_cost_ESS[44] + y_cost_ESS[45] + y_cost_ESS[46] + y_cost_ESS[47]
   + y_cost_ESS[48] + y_cost_ESS[49] + y_cost_ESS[50] + y_cost_ESS[51]
   + y_cost_ESS[52] + y_cost_ESS[53] + y_cost_ESS[54] + y_cost_ESS[55]
   + y_cost_ESS[56] + y_cost_ESS[57] + y_cost_ESS[58] + y_cost_ESS[59]
   + y_cost_ESS[60] + y_cost_ESS[61] + y_cost_ESS[62] + y_cost_ESS[63]
   + y_cost_ESS[64] + y_cost_ESS[65] + y_cost_ESS[66] + y_cost_ESS[67]
   + y_cost_ESS[68] + y_cost_ESS[69] + y_cost_ESS[70] + y_cost_ESS[71]
   + y_cost_ESS[72] + y_cost_ESS[73] + y_cost_ESS[74] + y_cost_ESS[75]
   + y_cost_ESS[76] + y_cost_ESS[77] + y_cost_ESS[78] + y_cost_ESS[79]
   + y_cost_ESS[80] + y_cost_ESS[81] + y_cost_ESS[82] + y_cost_ESS[83]
   + y_cost_ESS[84] + y_cost_ESS[85] + y_cost_ESS[86] + y_cost_ESS[87]
   + y_cost_ESS[88] + y_cost_ESS[89] + y_cost_ESS[90] + y_cost_ESS[91]
   + y_cost_ESS[92] + y_cost_ESS[93] + y_cost_ESS[94] + y_cost_ESS[95]
   + 0.00085683 simga_cut_n[0,1] + 0.00342732023208619 simga_cut_n[0,2]
   + 0.00771147052219392 simga_cut_n[0,3]
   + 0.0137092809283447 simga_cut_n[0,4]
   + 0.0214207514505387 simga_cut_n[0,5]
   + 0.0308458820887757 simga_cut_n[0,6]
   + 0.0419846728430558 simga_cut_n[0,7]
   + 0.054837123713379 simga_cut_n[0,8]
   + 0.0694032346997452 simga_cut_n[0,9] + 0.00085683 simga_cut_n[1,1]
   + 0.00342732023208619 simga_cut_n[1,2]
   + 0.00771147052219392 simga_cut_n[1,3]
   + 0.0137092809283447 simga_cut_n[1,4]
   + 0.0214207514505387 simga_cut_n[1,5]
   + 0.0308458820887757 simga_cut_n[1,6]
   + 0.0419846728430558 simga_cut_n[1,7]
   + 0.054837123713379 simga_cut_n[1,8]
   + 0.0694032346997452 simga_cut_n[1,9] + 0.000867 simga_cut_n[2,1]
   + 0.00346800019454956 simga_cut_n[2,2]
   + 0.00780300043773652 simga_cut_n[2,3]
   + 0.0138720007781983 simga_cut_n[2,4]
   + 0.0216750012159348 simga_cut_n[2,5]
   + 0.0312120017509461 simga_cut_n[2,6]
   + 0.0424830023832322 simga_cut_n[2,7]
   + 0.055488003112793 simga_cut_n[2,8]
   + 0.0702270039396286 simga_cut_n[2,9] + 0.000867 simga_cut_n[3,1]
   + 0.00346800019454956 simga_cut_n[3,2]
   + 0.00780300043773652 simga_cut_n[3,3]
   + 0.0138720007781983 simga_cut_n[3,4]
   + 0.0216750012159348 simga_cut_n[3,5]
   + 0.0312120017509461 simga_cut_n[3,6]
   + 0.0424830023832322 simga_cut_n[3,7]
   + 0.055488003112793 simga_cut_n[3,8]
   + 0.0702270039396286 simga_cut_n[3,9] + 0.000867 simga_cut_n[4,1]
   + 0.00346800019454956 simga_cut_n[4,2]
   + 0.00780300043773652 simga_cut_n[4,3]
   + 0.0138720007781983 simga_cut_n[4,4]
   + 0.0216750012159348 simga_cut_n[4,5]
   + 0.0312120017509461 simga_cut_n[4,6]
   + 0.0424830023832322 simga_cut_n[4,7]
   + 0.055488003112793 simga_cut_n[4,8]
   + 0.0702270039396286 simga_cut_n[4,9] + 0.00124848 simga_cut_n[5,1]
   + 0.00499391981323242 simga_cut_n[5,2]
   + 0.011236319579773 simga_cut_n[5,3]
   + 0.0199756792529297 simga_cut_n[5,4]
   + 0.0312119988327026 simga_cut_n[5,5]
   + 0.0449452783190918 simga_cut_n[5,6]
   + 0.0611755177120972 simga_cut_n[5,7]
   + 0.0799027170117188 simga_cut_n[5,8]
   + 0.1011268762179566 simga_cut_n[5,9] + 0.00124848 simga_cut_n[6,1]
   + 0.00499391981323242 simga_cut_n[6,2]
   + 0.011236319579773 simga_cut_n[6,3]
   + 0.0199756792529297 simga_cut_n[6,4]
   + 0.0312119988327026 simga_cut_n[6,5]
   + 0.0449452783190918 simga_cut_n[6,6]
   + 0.0611755177120972 simga_cut_n[6,7]
   + 0.0799027170117188 simga_cut_n[6,8]
   + 0.1011268762179566 simga_cut_n[6,9] + 0.00124848 simga_cut_n[7,1]
   + 0.00499391981323242 simga_cut_n[7,2]
   + 0.011236319579773 simga_cut_n[7,3]
   + 0.0199756792529297 simga_cut_n[7,4]
   + 0.0312119988327026 simga_cut_n[7,5]
   + 0.0449452783190918 simga_cut_n[7,6]
   + 0.0611755177120972 simga_cut_n[7,7]
   + 0.0799027170117188 simga_cut_n[7,8]
   + 0.1011268762179566 simga_cut_n[7,9] + 0.00124848 simga_cut_n[8,1]
   + 0.00499391981323242 simga_cut_n[8,2]
   + 0.011236319579773 simga_cut_n[8,3]
   + 0.0199756792529297 simga_cut_n[8,4]
   + 0.0312119988327026 simga_cut_n[8,5]
   + 0.0449452783190918 simga_cut_n[8,6]
   + 0.0611755177120972 simga_cut_n[8,7]
   + 0.0799027170117188 simga_cut_n[8,8]
   + 0.1011268762179566 simga_cut_n[8,9] + 0.00124848 simga_cut_n[9,1]
   + 0.00499391981323242 simga_cut_n[9,2]
   + 0.011236319579773 simga_cut_n[9,3]
   + 0.0199756792529297 simga_cut_n[9,4]
   + 0.0312119988327026 simga_cut_n[9,5]
   + 0.0449452783190918 simga_cut_n[9,6]
   + 0.0611755177120972 simga_cut_n[9,7]
   + 0.0799027170117188 simga_cut_n[9,8]
   + 0.1011268762179566 simga_cut_n[9,9]
   + 0.00169932016342163 simga_cut_n[10,1]
   + 0.00679728065368654 simga_cut_n[10,2]
   + 0.0152938814707947 simga_cut_n[10,3]
   + 0.0271891226147462 simga_cut_n[10,4]
   + 0.0424830040855409 simga_cut_n[10,5]
   + 0.0611755258831789 simga_cut_n[10,6]
   + 0.0832666880076601 simga_cut_n[10,7]
   + 0.1087564904589846 simga_cut_n[10,8]
   + 0.1376449332371524 simga_cut_n[10,9]
   + 0.00169932016342163 simga_cut_n[11,1]
   + 0.00679728065368654 simga_cut_n[11,2]
   + 0.0152938814707947 simga_cut_n[11,3]
   + 0.0271891226147462 simga_cut_n[11,4]
   + 0.0424830040855409 simga_cut_n[11,5]
   + 0.0611755258831789 simga_cut_n[11,6]
   + 0.0832666880076601 simga_cut_n[11,7]
   + 0.1087564904589846 simga_cut_n[11,8]
   + 0.1376449332371524 simga_cut_n[11,9]
   + 0.00168506983726502 simga_cut_n[12,1]
   + 0.00674027934906007 simga_cut_n[12,2]
   + 0.0151656285353852 simga_cut_n[12,3]
   + 0.0269611173962403 simga_cut_n[12,4]
   + 0.0421267459316255 simga_cut_n[12,5]
   + 0.0606625141415407 simga_cut_n[12,6]
   + 0.0825684220259859 simga_cut_n[12,7]
   + 0.1078444695849612 simga_cut_n[12,8]
   + 0.1364906568184665 simga_cut_n[12,9]
   + 0.00168506983726502 simga_cut_n[13,1]
   + 0.00674027934906007 simga_cut_n[13,2]
   + 0.0151656285353852 simga_cut_n[13,3]
   + 0.0269611173962403 simga_cut_n[13,4]
   + 0.0421267459316255 simga_cut_n[13,5]
   + 0.0606625141415407 simga_cut_n[13,6]
   + 0.0825684220259859 simga_cut_n[13,7]
   + 0.1078444695849612 simga_cut_n[13,8]
   + 0.1364906568184665 simga_cut_n[13,9]
   + 0.00168506983726502 simga_cut_n[14,1]
   + 0.00674027934906007 simga_cut_n[14,2]
   + 0.0151656285353852 simga_cut_n[14,3]
   + 0.0269611173962403 simga_cut_n[14,4]
   + 0.0421267459316255 simga_cut_n[14,5]
   + 0.0606625141415407 simga_cut_n[14,6]
   + 0.0825684220259859 simga_cut_n[14,7]
   + 0.1078444695849612 simga_cut_n[14,8]
   + 0.1364906568184665 simga_cut_n[14,9] + 0.00220323 simga_cut_n[15,1]
   + 0.00881292024810791 simga_cut_n[15,2]
   + 0.0198290705582428 simga_cut_n[15,3]
   + 0.0352516809924316 simga_cut_n[15,4]
   + 0.0550807515506744 simga_cut_n[15,5]
   + 0.0793162822329712 simga_cut_n[15,6]
   + 0.1079582730393219 simga_cut_n[15,7]
   + 0.1410067239697265 simga_cut_n[15,8]
   + 0.1784616350241852 simga_cut_n[15,9] + 0.00220323 simga_cut_n[16,1]
   + 0.00881292024810791 simga_cut_n[16,2]
   + 0.0198290705582428 simga_cut_n[16,3]
   + 0.0352516809924316 simga_cut_n[16,4]
   + 0.0550807515506744 simga_cut_n[16,5]
   + 0.0793162822329712 simga_cut_n[16,6]
   + 0.1079582730393219 simga_cut_n[16,7]
   + 0.1410067239697265 simga_cut_n[16,8]
   + 0.1784616350241852 simga_cut_n[16,9] + 0.00220323 simga_cut_n[17,1]
   + 0.00881292024810791 simga_cut_n[17,2]
   + 0.0198290705582428 simga_cut_n[17,3]
   + 0.0352516809924316 simga_cut_n[17,4]
   + 0.0550807515506744 simga_cut_n[17,5]
   + 0.0793162822329712 simga_cut_n[17,6]
   + 0.1079582730393219 simga_cut_n[17,7]
   + 0.1410067239697265 simga_cut_n[17,8]
   + 0.1784616350241852 simga_cut_n[17,9] + 0.00220323 simga_cut_n[18,1]
   + 0.00881292024810791 simga_cut_n[18,2]
   + 0.0198290705582428 simga_cut_n[18,3]
   + 0.0352516809924316 simga_cut_n[18,4]
   + 0.0550807515506744 simga_cut_n[18,5]
   + 0.0793162822329712 simga_cut_n[18,6]
   + 0.1079582730393219 simga_cut_n[18,7]
   + 0.1410067239697265 simga_cut_n[18,8]
   + 0.1784616350241852 simga_cut_n[18,9] + 0.00220323 simga_cut_n[19,1]
   + 0.00881292024810791 simga_cut_n[19,2]
   + 0.0198290705582428 simga_cut_n[19,3]
   + 0.0352516809924316 simga_cut_n[19,4]
   + 0.0550807515506744 simga_cut_n[19,5]
   + 0.0793162822329712 simga_cut_n[19,6]
   + 0.1079582730393219 simga_cut_n[19,7]
   + 0.1410067239697265 simga_cut_n[19,8]
   + 0.1784616350241852 simga_cut_n[19,9] + 0.00279075 simga_cut_n[20,1]
   + 0.0111629996509552 simga_cut_n[20,2]
   + 0.0251167492146492 simga_cut_n[20,3]
   + 0.0446519986038208 simga_cut_n[20,4]
   + 0.06976874781847 simga_cut_n[20,5]
   + 0.1004669968585968 simga_cut_n[20,6]
   + 0.1367467457242013 simga_cut_n[20,7]
   + 0.1786079944152832 simga_cut_n[20,8]
   + 0.2260507429318429 simga_cut_n[20,9] + 0.00279075 simga_cut_n[21,1]
   + 0.0111629996509552 simga_cut_n[21,2]
   + 0.0251167492146492 simga_cut_n[21,3]
   + 0.0446519986038208 simga_cut_n[21,4]
   + 0.06976874781847 simga_cut_n[21,5]
   + 0.1004669968585968 simga_cut_n[21,6]
   + 0.1367467457242013 simga_cut_n[21,7]
   + 0.1786079944152832 simga_cut_n[21,8]
   + 0.2260507429318429 simga_cut_n[21,9] + 0.00279075 simga_cut_n[22,1]
   + 0.0111629996509552 simga_cut_n[22,2]
   + 0.0251167492146492 simga_cut_n[22,3]
   + 0.0446519986038208 simga_cut_n[22,4]
   + 0.06976874781847 simga_cut_n[22,5]
   + 0.1004669968585968 simga_cut_n[22,6]
   + 0.1367467457242013 simga_cut_n[22,7]
   + 0.1786079944152832 simga_cut_n[22,8]
   + 0.2260507429318429 simga_cut_n[22,9] + 0.00279075 simga_cut_n[23,1]
   + 0.0111629996509552 simga_cut_n[23,2]
   + 0.0251167492146492 simga_cut_n[23,3]
   + 0.0446519986038208 simga_cut_n[23,4]
   + 0.06976874781847 simga_cut_n[23,5]
   + 0.1004669968585968 simga_cut_n[23,6]
   + 0.1367467457242013 simga_cut_n[23,7]
   + 0.1786079944152832 simga_cut_n[23,8]
   + 0.2260507429318429 simga_cut_n[23,9] + 0.00279075 simga_cut_n[24,1]
   + 0.0111629996509552 simga_cut_n[24,2]
   + 0.0251167492146492 simga_cut_n[24,3]
   + 0.0446519986038208 simga_cut_n[24,4]
   + 0.06976874781847 simga_cut_n[24,5]
   + 0.1004669968585968 simga_cut_n[24,6]
   + 0.1367467457242013 simga_cut_n[24,7]
   + 0.1786079944152832 simga_cut_n[24,8]
   + 0.2260507429318429 simga_cut_n[24,9]
   + 0.0034476302133751 simga_cut_n[25,1]
   + 0.0137905208535004 simga_cut_n[25,2]
   + 0.0310286719203759 simga_cut_n[25,3]
   + 0.0551620834140015 simga_cut_n[25,4]
   + 0.0861907553343774 simga_cut_n[25,5]
   + 0.1241146876815034 simga_cut_n[25,6]
   + 0.1689338804553796 simga_cut_n[25,7]
   + 0.2206483336560061 simga_cut_n[25,8]
   + 0.2792580472833827 simga_cut_n[25,9] + 0.00279075 simga_cut_n[26,1]
   + 0.0111629996509552 simga_cut_n[26,2]
   + 0.0251167492146492 simga_cut_n[26,3]
   + 0.0446519986038208 simga_cut_n[26,4]
   + 0.06976874781847 simga_cut_n[26,5]
   + 0.1004669968585968 simga_cut_n[26,6]
   + 0.1367467457242013 simga_cut_n[26,7]
   + 0.1786079944152832 simga_cut_n[26,8]
   + 0.2260507429318429 simga_cut_n[26,9]
   + 0.0376319987182617 simga_cut_n[27,1]
   + 0.1505279948730469 simga_cut_n[27,2]
   + 0.3386879884643555 simga_cut_n[27,3]
   + 0.6021119794921878 simga_cut_n[27,4]
   + 0.9407999679565432 simga_cut_n[27,5]
   + 1.354751953857422 simga_cut_n[27,6]
   + 1.843967937194825 simga_cut_n[27,7]
   + 2.408447917968751 simga_cut_n[27,8]
   + 3.0481918961792 simga_cut_n[27,9] + 0.12301875 simga_cut_n[28,1]
   + 0.492075 simga_cut_n[28,2] + 1.10716875 simga_cut_n[28,3]
   + 1.9683 simga_cut_n[28,4] + 3.07546875 simga_cut_n[28,5]
   + 4.428675 simga_cut_n[28,6] + 6.02791875 simga_cut_n[28,7]
   + 7.8732 simga_cut_n[28,8] + 9.964518750000002 simga_cut_n[28,9]
   + 0.3122122475115975 simga_cut_n[29,1]
   + 1.24884899004639 simga_cut_n[29,2]
   + 2.809910227604377 simga_cut_n[29,3]
   + 4.995395960185561 simga_cut_n[29,4]
   + 7.805306187789938 simga_cut_n[29,5]
   + 11.23964091041751 simga_cut_n[29,6]
   + 15.29840012806828 simga_cut_n[29,7]
   + 19.98158384074224 simga_cut_n[29,8]
   + 25.2891920484394 simga_cut_n[29,9]
   + 0.1732322799000551 simga_cut_n[30,1]
   + 0.6929291196002204 simga_cut_n[30,2]
   + 1.559090519100496 simga_cut_n[30,3]
   + 2.771716478400882 simga_cut_n[30,4]
   + 4.330806997501377 simga_cut_n[30,5]
   + 6.236362076401982 simga_cut_n[30,6]
   + 8.488381715102696 simga_cut_n[30,7]
   + 11.08686591360353 simga_cut_n[30,8]
   + 14.03181467190446 simga_cut_n[30,9]
   + 1.119596435592957 simga_cut_n[31,1]
   + 4.478385742371827 simga_cut_n[31,2]
   + 10.07636792033661 simga_cut_n[31,3]
   + 17.91354296948731 simga_cut_n[31,4]
   + 27.98991088982391 simga_cut_n[31,5]
   + 40.30547168134643 simga_cut_n[31,6]
   + 54.86022534405488 simga_cut_n[31,7]
   + 71.65417187794922 simga_cut_n[31,8]
   + 90.68731128302949 simga_cut_n[31,9]
   + 3.025855488389282 simga_cut_n[32,1]
   + 12.10342195355713 simga_cut_n[32,2]
   + 27.23269939550354 simga_cut_n[32,3]
   + 48.41368781422852 simga_cut_n[32,4]
   + 75.64638720973205 simga_cut_n[32,5]
   + 108.9307975820142 simga_cut_n[32,6]
   + 148.2669189310748 simga_cut_n[32,7]
   + 193.6547512569141 simga_cut_n[32,8]
   + 245.0942945595319 simga_cut_n[32,9]
   + 5.455357243835461 simga_cut_n[33,1]
   + 21.82142897534184 simga_cut_n[33,2]
   + 49.09821519451914 simga_cut_n[33,3]
   + 87.28571590136737 simga_cut_n[33,4]
   + 136.3839310958865 simga_cut_n[33,5]
   + 196.3928607780766 simga_cut_n[33,6]
   + 267.3125049479376 simga_cut_n[33,7]
   + 349.1428636054695 simga_cut_n[33,8]
   + 441.8839367506723 simga_cut_n[33,9]
   + 7.181483633320313 simga_cut_n[34,1]
   + 28.72593453328125 simga_cut_n[34,2]
   + 64.63335269988282 simga_cut_n[34,3]
   + 114.903738133125 simga_cut_n[34,4]
   + 179.5370908330078 simga_cut_n[34,5]
   + 258.5334107995313 simga_cut_n[34,6]
   + 351.8926980326953 simga_cut_n[34,7] + 459.6149525325 simga_cut_n[34,8]
   + 581.7001742989453 simga_cut_n[34,9]
   + 7.972887875946889 simga_cut_n[35,1]
   + 31.89155150378756 simga_cut_n[35,2]
   + 71.75599088352202 simga_cut_n[35,3]
   + 127.5662060151502 simga_cut_n[35,4]
   + 199.3221968986722 simga_cut_n[35,5]
   + 287.0239635340881 simga_cut_n[35,6]
   + 390.6715059213976 simga_cut_n[35,7]
   + 510.2648240606009 simga_cut_n[35,8]
   + 645.8039179516981 simga_cut_n[35,9]
   + 9.377363874335121 simga_cut_n[36,1]
   + 37.50945549734048 simga_cut_n[36,2]
   + 84.39627486901608 simga_cut_n[36,3]
   + 150.0378219893619 simga_cut_n[36,4]
   + 234.434096858378 simga_cut_n[36,5]
   + 337.5850994760643 simga_cut_n[36,6]
   + 459.4908298424209 simga_cut_n[36,7]
   + 600.1512879574477 simga_cut_n[36,8]
   + 759.5664738211447 simga_cut_n[36,9]
   + 9.885239141825283 simga_cut_n[37,1]
   + 39.54095656730113 simga_cut_n[37,2]
   + 88.96715227642756 simga_cut_n[37,3]
   + 158.1638262692045 simga_cut_n[37,4]
   + 247.1309785456321 simga_cut_n[37,5]
   + 355.8686091057102 simga_cut_n[37,6]
   + 484.376717949439 simga_cut_n[37,7]
   + 632.6553050768181 simga_cut_n[37,8]
   + 800.704370487848 simga_cut_n[37,9]
   + 10.62260222416567 simga_cut_n[38,1]
   + 42.49040889666269 simga_cut_n[38,2]
   + 95.60342001749103 simga_cut_n[38,3]
   + 169.9616355866507 simga_cut_n[38,4]
   + 265.5650556041418 simga_cut_n[38,5]
   + 382.4136800699641 simga_cut_n[38,6]
   + 520.507508984118 simga_cut_n[38,7]
   + 679.846542346603 simga_cut_n[38,8]
   + 860.4307801574191 simga_cut_n[38,9]
   + 9.622731121752782 simga_cut_n[39,1]
   + 38.49092448701113 simga_cut_n[39,2]
   + 86.60458009577503 simga_cut_n[39,3]
   + 153.9636979480445 simga_cut_n[39,4]
   + 240.5682780438195 simga_cut_n[39,5]
   + 346.4183203831001 simga_cut_n[39,6]
   + 471.5138249658864 simga_cut_n[39,7]
   + 615.854791792178 simga_cut_n[39,8]
   + 779.4412208619752 simga_cut_n[39,9]
   + 14.10951848683303 simga_cut_n[40,1]
   + 56.43807394733211 simga_cut_n[40,2]
   + 126.9856663814972 simga_cut_n[40,3]
   + 225.7522957893284 simga_cut_n[40,4]
   + 352.7379621708256 simga_cut_n[40,5]
   + 507.9426655259888 simga_cut_n[40,6]
   + 691.3664058548181 simga_cut_n[40,7]
   + 903.0091831573137 simga_cut_n[40,8]
   + 1142.870997433475 simga_cut_n[40,9]
   + 15.92208297968593 simga_cut_n[41,1]
   + 63.68833191874372 simga_cut_n[41,2]
   + 143.2987468171734 simga_cut_n[41,3]
   + 254.7533276749749 simga_cut_n[41,4]
   + 398.0520744921483 simga_cut_n[41,5]
   + 573.1949872686935 simga_cut_n[41,6]
   + 780.1820660046104 simga_cut_n[41,7]
   + 1019.0133106999 simga_cut_n[41,8] + 1289.68872135456 simga_cut_n[41,9]
   + 8.146668013163723 simga_cut_n[42,1]
   + 32.58667205265489 simga_cut_n[42,2]
   + 73.32001211847353 simga_cut_n[42,3]
   + 130.3466882106196 simga_cut_n[42,4]
   + 203.6667003290931 simga_cut_n[42,5]
   + 293.2800484738941 simga_cut_n[42,6]
   + 399.1867326450225 simga_cut_n[42,7]
   + 521.3867528424782 simga_cut_n[42,8]
   + 659.8801090662616 simga_cut_n[42,9]
   + 9.106431128954593 simga_cut_n[43,1]
   + 36.42572451581837 simga_cut_n[43,2]
   + 81.95788016059133 simga_cut_n[43,3]
   + 145.7028980632735 simga_cut_n[43,4]
   + 227.6607782238648 simga_cut_n[43,5]
   + 327.8315206423653 simga_cut_n[43,6]
   + 446.2151253187751 simga_cut_n[43,7]
   + 582.811592253094 simga_cut_n[43,8]
   + 737.620921445322 simga_cut_n[43,9]
   + 16.92496626529292 simga_cut_n[44,1]
   + 67.69986506117169 simga_cut_n[44,2]
   + 152.3246963876363 simga_cut_n[44,3]
   + 270.7994602446868 simga_cut_n[44,4]
   + 423.124156632323 simga_cut_n[44,5]
   + 609.2987855505452 simga_cut_n[44,6]
   + 829.3233469993531 simga_cut_n[44,7]
   + 1083.197840978747 simga_cut_n[44,8]
   + 1370.922267488726 simga_cut_n[44,9]
   + 18.00132181411051 simga_cut_n[45,1]
   + 72.00528725644205 simga_cut_n[45,2]
   + 162.0118963269946 simga_cut_n[45,3]
   + 288.0211490257682 simga_cut_n[45,4]
   + 450.0330453527627 simga_cut_n[45,5]
   + 648.0475853079784 simga_cut_n[45,6]
   + 882.0647688914148 simga_cut_n[45,7]
   + 1152.084596103073 simga_cut_n[45,8]
   + 1458.107066942951 simga_cut_n[45,9]
   + 24.17324910699454 simga_cut_n[46,1]
   + 96.69299642797814 simga_cut_n[46,2]
   + 217.5592419629508 simga_cut_n[46,3]
   + 386.7719857119126 simga_cut_n[46,4]
   + 604.3312276748633 simga_cut_n[46,5]
   + 870.2369678518032 simga_cut_n[46,6]
   + 1184.489206242732 simga_cut_n[46,7]
   + 1547.08794284765 simga_cut_n[46,8]
   + 1958.033177666557 simga_cut_n[46,9]
   + 24.91191318593543 simga_cut_n[47,1]
   + 99.64765274374173 simga_cut_n[47,2]
   + 224.2072186734189 simga_cut_n[47,3]
   + 398.5906109749669 simga_cut_n[47,4]
   + 622.7978296483858 simga_cut_n[47,5]
   + 896.8288746936756 simga_cut_n[47,6]
   + 1220.683746110836 simga_cut_n[47,7]
   + 1594.362443899868 simga_cut_n[47,8]
   + 2017.864968060769 simga_cut_n[47,9]
   + 27.59518083305004 simga_cut_n[48,1]
   + 110.3807233322002 simga_cut_n[48,2]
   + 248.3566274974504 simga_cut_n[48,3]
   + 441.5228933288007 simga_cut_n[48,4]
   + 689.8795208262511 simga_cut_n[48,5]
   + 993.4265099898015 simga_cut_n[48,6]
   + 1352.163860819452 simga_cut_n[48,7]
   + 1766.091573315203 simga_cut_n[48,8]
   + 2235.209647477053 simga_cut_n[48,9]
   + 30.96526913757961 simga_cut_n[49,1]
   + 123.8610765503185 simga_cut_n[49,2]
   + 278.6874222382166 simga_cut_n[49,3]
   + 495.4443062012738 simga_cut_n[49,4]
   + 774.1317284394905 simga_cut_n[49,5]
   + 1114.749688952867 simga_cut_n[49,6]
   + 1517.298187741401 simga_cut_n[49,7]
   + 1981.777224805095 simga_cut_n[49,8]
   + 2508.18680014395 simga_cut_n[49,9]
   + 27.83787876785119 simga_cut_n[50,1]
   + 111.3515150714047 simga_cut_n[50,2]
   + 250.5409089106607 simga_cut_n[50,3]
   + 445.406060285619 simga_cut_n[50,4]
   + 695.9469691962797 simga_cut_n[50,5]
   + 1002.163635642643 simga_cut_n[50,6]
   + 1364.056059624708 simga_cut_n[50,7]
   + 1781.624241142476 simga_cut_n[50,8]
   + 2254.868180195946 simga_cut_n[50,9]
   + 28.66393681460079 simga_cut_n[51,1]
   + 114.6557472584032 simga_cut_n[51,2]
   + 257.9754313314071 simga_cut_n[51,3]
   + 458.6229890336126 simga_cut_n[51,4]
   + 716.5984203650198 simga_cut_n[51,5]
   + 1031.901725325628 simga_cut_n[51,6]
   + 1404.532903915439 simga_cut_n[51,7]
   + 1834.49195613445 simga_cut_n[51,8]
   + 2321.778881982664 simga_cut_n[51,9]
   + 29.09694733182307 simga_cut_n[52,1]
   + 116.3877893272923 simga_cut_n[52,2]
   + 261.8725259864077 simga_cut_n[52,3]
   + 465.5511573091691 simga_cut_n[52,4]
   + 727.4236832955767 simga_cut_n[52,5]
   + 1047.490103945631 simga_cut_n[52,6]
   + 1425.75041925933 simga_cut_n[52,7]
   + 1862.204629236676 simga_cut_n[52,8]
   + 2356.852733877668 simga_cut_n[52,9]
   + 28.54627334340142 simga_cut_n[53,1]
   + 114.1850933736057 simga_cut_n[53,2]
   + 256.9164600906127 simga_cut_n[53,3]
   + 456.7403734944227 simga_cut_n[53,4]
   + 713.6568335850354 simga_cut_n[53,5]
   + 1027.665840362451 simga_cut_n[53,6]
   + 1398.76739382667 simga_cut_n[53,7]
   + 1826.961493977691 simga_cut_n[53,8]
   + 2312.248140815515 simga_cut_n[53,9]
   + 29.6528844107589 simga_cut_n[54,1]
   + 118.6115376430356 simga_cut_n[54,2]
   + 266.8759596968301 simga_cut_n[54,3]
   + 474.4461505721424 simga_cut_n[54,4]
   + 741.3221102689724 simga_cut_n[54,5]
   + 1067.503838787321 simga_cut_n[54,6]
   + 1452.991336127186 simga_cut_n[54,7]
   + 1897.784602288569 simga_cut_n[54,8]
   + 2401.88363727147 simga_cut_n[54,9]
   + 26.68489688268196 simga_cut_n[55,1]
   + 106.7395875307278 simga_cut_n[55,2]
   + 240.1640719441376 simga_cut_n[55,3]
   + 426.9583501229113 simga_cut_n[55,4]
   + 667.1224220670489 simga_cut_n[55,5]
   + 960.6562877765505 simga_cut_n[55,6]
   + 1307.559947251416 simga_cut_n[55,7]
   + 1707.833400491645 simga_cut_n[55,8]
   + 2161.476647497238 simga_cut_n[55,9]
   + 19.17740084548896 simga_cut_n[56,1]
   + 76.70960338195584 simga_cut_n[56,2]
   + 172.5966076094006 simga_cut_n[56,3]
   + 306.8384135278234 simga_cut_n[56,4]
   + 479.435021137224 simga_cut_n[56,5]
   + 690.3864304376026 simga_cut_n[56,6]
   + 939.6926414289591 simga_cut_n[56,7]
   + 1227.353654111294 simga_cut_n[56,8]
   + 1553.369468484605 simga_cut_n[56,9]
   + 20.25919565582528 simga_cut_n[57,1]
   + 81.03678262330111 simga_cut_n[57,2]
   + 182.3327609024275 simga_cut_n[57,3]
   + 324.1471304932044 simga_cut_n[57,4]
   + 506.479891395632 simga_cut_n[57,5] + 729.33104360971 simga_cut_n[57,6]
   + 992.7005871354385 simga_cut_n[57,7]
   + 1296.588521972818 simga_cut_n[57,8]
   + 1640.994848121847 simga_cut_n[57,9]
   + 19.5835543495283 simga_cut_n[58,1]
   + 78.3342173981132 simga_cut_n[58,2]
   + 176.2519891457547 simga_cut_n[58,3]
   + 313.3368695924528 simga_cut_n[58,4]
   + 489.5888587382075 simga_cut_n[58,5]
   + 705.0079565830187 simga_cut_n[58,6]
   + 959.5941631268867 simga_cut_n[58,7]
   + 1253.347478369811 simga_cut_n[58,8]
   + 1586.267902311793 simga_cut_n[58,9]
   + 22.58969402306096 simga_cut_n[59,1]
   + 90.35877609224386 simga_cut_n[59,2]
   + 203.3072462075487 simga_cut_n[59,3]
   + 361.4351043689754 simga_cut_n[59,4]
   + 564.742350576524 simga_cut_n[59,5]
   + 813.2289848301947 simga_cut_n[59,6]
   + 1106.895007129987 simga_cut_n[59,7]
   + 1445.740417475902 simga_cut_n[59,8]
   + 1829.765215867938 simga_cut_n[59,9]
   + 18.63949523675176 simga_cut_n[60,1]
   + 74.55798094700704 simga_cut_n[60,2]
   + 167.7554571307658 simga_cut_n[60,3]
   + 298.2319237880282 simga_cut_n[60,4]
   + 465.987380918794 simga_cut_n[60,5]
   + 671.0218285230633 simga_cut_n[60,6]
   + 913.3352666008361 simga_cut_n[60,7]
   + 1192.927695152113 simga_cut_n[60,8]
   + 1509.799114176893 simga_cut_n[60,9]
   + 16.15482745575442 simga_cut_n[61,1]
   + 64.61930982301769 simga_cut_n[61,2]
   + 145.3934471017897 simga_cut_n[61,3]
   + 258.4772392920707 simga_cut_n[61,4]
   + 403.8706863938605 simga_cut_n[61,5]
   + 581.573788407159 simga_cut_n[61,6]
   + 791.5865453319666 simga_cut_n[61,7]
   + 1033.908957168283 simga_cut_n[61,8]
   + 1308.541023916108 simga_cut_n[61,9]
   + 10.32609721442094 simga_cut_n[62,1]
   + 41.30438885768375 simga_cut_n[62,2]
   + 92.93487492978845 simga_cut_n[62,3]
   + 165.217555430735 simga_cut_n[62,4]
   + 258.1524303605235 simga_cut_n[62,5]
   + 371.7394997191538 simga_cut_n[62,6]
   + 505.978763506626 simga_cut_n[62,7]
   + 660.8702217229401 simga_cut_n[62,8]
   + 836.4138743680961 simga_cut_n[62,9]
   + 18.92912958305391 simga_cut_n[63,1]
   + 75.71651833221564 simga_cut_n[63,2]
   + 170.3621662474852 simga_cut_n[63,3]
   + 302.8660733288626 simga_cut_n[63,4]
   + 473.2282395763476 simga_cut_n[63,5]
   + 681.4486649899407 simga_cut_n[63,6]
   + 927.5273495696416 simga_cut_n[63,7]
   + 1211.46429331545 simga_cut_n[63,8]
   + 1533.259496227367 simga_cut_n[63,9]
   + 11.01373505630766 simga_cut_n[64,1]
   + 44.05494022523066 simga_cut_n[64,2]
   + 99.12361550676899 simga_cut_n[64,3]
   + 176.2197609009226 simga_cut_n[64,4]
   + 275.3433764076916 simga_cut_n[64,5]
   + 396.4944620270759 simga_cut_n[64,6]
   + 539.6730177590755 simga_cut_n[64,7]
   + 704.8790436036905 simga_cut_n[64,8]
   + 892.1125395609208 simga_cut_n[64,9]
   + 7.827458556285715 simga_cut_n[65,1]
   + 31.30983422514286 simga_cut_n[65,2]
   + 70.44712700657146 simga_cut_n[65,3]
   + 125.2393369005714 simga_cut_n[65,4]
   + 195.686463907143 simga_cut_n[65,5]
   + 281.7885080262859 simga_cut_n[65,6]
   + 383.5454692580003 simga_cut_n[65,7]
   + 500.9573476022858 simga_cut_n[65,8]
   + 634.0241430591431 simga_cut_n[65,9]
   + 5.214154980477581 simga_cut_n[66,1]
   + 20.85661992191032 simga_cut_n[66,2]
   + 46.92739482429823 simga_cut_n[66,3]
   + 83.42647968764129 simga_cut_n[66,4]
   + 130.3538745119395 simga_cut_n[66,5]
   + 187.7095792971929 simga_cut_n[66,6]
   + 255.4935940434015 simga_cut_n[66,7]
   + 333.7059187505652 simga_cut_n[66,8]
   + 422.346553418684 simga_cut_n[66,9]
   + 16.60670906289656 simga_cut_n[67,1]
   + 66.42683625158622 simga_cut_n[67,2]
   + 149.460381566069 simga_cut_n[67,3]
   + 265.7073450063449 simga_cut_n[67,4]
   + 415.1677265724138 simga_cut_n[67,5]
   + 597.841526264276 simga_cut_n[67,6]
   + 813.7287440819313 simga_cut_n[67,7]
   + 1062.82938002538 simga_cut_n[67,8]
   + 1345.143434094621 simga_cut_n[67,9]
   + 3.433416335474857 simga_cut_n[68,1]
   + 13.73366534189943 simga_cut_n[68,2]
   + 30.90074701927371 simga_cut_n[68,3]
   + 54.93466136759771 simga_cut_n[68,4]
   + 85.83540838687142 simga_cut_n[68,5]
   + 123.6029880770948 simga_cut_n[68,6]
   + 168.237400438268 simga_cut_n[68,7]
   + 219.7386454703908 simga_cut_n[68,8]
   + 278.1067231734634 simga_cut_n[68,9]
   + 4.28049053128052 simga_cut_n[69,1]
   + 17.12196212512208 simga_cut_n[69,2]
   + 38.52441478152468 simga_cut_n[69,3]
   + 68.48784850048833 simga_cut_n[69,4]
   + 107.012263282013 simga_cut_n[69,5]
   + 154.0976591260987 simga_cut_n[69,6]
   + 209.7440360327455 simga_cut_n[69,7]
   + 273.9513940019533 simga_cut_n[69,8]
   + 346.7197330337222 simga_cut_n[69,9]
   + 0.901240875162049 simga_cut_n[70,1]
   + 3.604963500648196 simga_cut_n[70,2]
   + 8.111167876458442 simga_cut_n[70,3]
   + 14.41985400259278 simga_cut_n[70,4]
   + 22.53102187905122 simga_cut_n[70,5]
   + 32.44467150583377 simga_cut_n[70,6]
   + 44.16080288294039 simga_cut_n[70,7]
   + 57.67941601037113 simga_cut_n[70,8]
   + 73.00051088812596 simga_cut_n[70,9]
   + 1.111181902287598 simga_cut_n[71,1]
   + 4.44472760915039 simga_cut_n[71,2]
   + 10.00063712058838 simga_cut_n[71,3]
   + 17.77891043660156 simga_cut_n[71,4]
   + 27.77954755718994 simga_cut_n[71,5]
   + 40.00254848235353 simga_cut_n[71,6]
   + 54.4479132120923 simga_cut_n[71,7]
   + 71.11564174640624 simga_cut_n[71,8]
   + 90.00573408529542 simga_cut_n[71,9]
   + 1.246140631988528 simga_cut_n[72,1]
   + 4.984562527954113 simga_cut_n[72,2]
   + 11.21526568789675 simga_cut_n[72,3]
   + 19.93825011181645 simga_cut_n[72,4]
   + 31.1535157997132 simga_cut_n[72,5]
   + 44.86106275158701 simga_cut_n[72,6]
   + 61.06089096743788 simga_cut_n[72,7]
   + 79.7530004472658 simga_cut_n[72,8]
   + 100.9373911910708 simga_cut_n[72,9]
   + 1.121429857609863 simga_cut_n[73,1]
   + 4.485719430439453 simga_cut_n[73,2]
   + 10.09286871848877 simga_cut_n[73,3]
   + 17.94287772175781 simga_cut_n[73,4]
   + 28.03574644024658 simga_cut_n[73,5]
   + 40.37147487395509 simga_cut_n[73,6]
   + 54.95006302288331 simga_cut_n[73,7]
   + 71.77151088703124 simga_cut_n[73,8]
   + 90.83581846639892 simga_cut_n[73,9]
   + 0.2940948199671937 simga_cut_n[74,1]
   + 1.176379279868775 simga_cut_n[74,2]
   + 2.646853379704743 simga_cut_n[74,3]
   + 4.705517119475099 simga_cut_n[74,4]
   + 7.352370499179842 simga_cut_n[74,5]
   + 10.58741351881897 simga_cut_n[74,6]
   + 14.41064617839249 simga_cut_n[74,7]
   + 18.82206847790039 simga_cut_n[74,8]
   + 23.82168041734269 simga_cut_n[74,9]
   + 0.0442867472190857 simga_cut_n[75,1]
   + 0.1771469888763429 simga_cut_n[75,2]
   + 0.3985807249717716 simga_cut_n[75,3]
   + 0.7085879555053718 simga_cut_n[75,4]
   + 1.107168680477143 simga_cut_n[75,5]
   + 1.594322899887086 simga_cut_n[75,6]
   + 2.170050613735201 simga_cut_n[75,7]
   + 2.834351822021487 simga_cut_n[75,8]
   + 3.587226524745944 simga_cut_n[75,9]
   + 0.00665523010780335 simga_cut_n[76,1]
   + 0.0266209204312134 simga_cut_n[76,2]
   + 0.0598970709702301 simga_cut_n[76,3]
   + 0.1064836817248535 simga_cut_n[76,4]
   + 0.1663807526950836 simga_cut_n[76,5]
   + 0.2395882838809204 simga_cut_n[76,6]
   + 0.3261062752823639 simga_cut_n[76,7]
   + 0.4259347268994141 simga_cut_n[76,8]
   + 0.5390736387320709 simga_cut_n[76,9] + 3.888e-05 simga_cut_n[77,1]
   + 0.00015552 simga_cut_n[77,2] + 0.00034992 simga_cut_n[77,3]
   + 0.00062208 simga_cut_n[77,4] + 0.000972 simga_cut_n[77,5]
   + 0.00139968011123657 simga_cut_n[77,6]
   + 0.00190512015140534 simga_cut_n[77,7]
   + 0.00248832019775391 simga_cut_n[77,8]
   + 0.00314928025028229 simga_cut_n[77,9] + 3.888e-05 simga_cut_n[78,1]
   + 0.00015552 simga_cut_n[78,2] + 0.00034992 simga_cut_n[78,3]
   + 0.00062208 simga_cut_n[78,4] + 0.000972 simga_cut_n[78,5]
   + 0.00139968011123657 simga_cut_n[78,6]
   + 0.00190512015140534 simga_cut_n[78,7]
   + 0.00248832019775391 simga_cut_n[78,8]
   + 0.00314928025028229 simga_cut_n[78,9] + 3.888e-05 simga_cut_n[79,1]
   + 0.00015552 simga_cut_n[79,2] + 0.00034992 simga_cut_n[79,3]
   + 0.00062208 simga_cut_n[79,4] + 0.000972 simga_cut_n[79,5]
   + 0.00139968011123657 simga_cut_n[79,6]
   + 0.00190512015140534 simga_cut_n[79,7]
   + 0.00248832019775391 simga_cut_n[79,8]
   + 0.00314928025028229 simga_cut_n[79,9] + 0.00015552 simga_cut_n[80,1]
   + 0.00062208 simga_cut_n[80,2] + 0.00139968011123657 simga_cut_n[80,3]
   + 0.00248832019775391 simga_cut_n[80,4]
   + 0.00388800030899048 simga_cut_n[80,5]
   + 0.0055987204449463 simga_cut_n[80,6]
   + 0.00762048060562135 simga_cut_n[80,7]
   + 0.00995328079101564 simga_cut_n[80,8]
   + 0.0125971210011292 simga_cut_n[80,9] + 0.00015552 simga_cut_n[81,1]
   + 0.00062208 simga_cut_n[81,2] + 0.00139968011123657 simga_cut_n[81,3]
   + 0.00248832019775391 simga_cut_n[81,4]
   + 0.00388800030899048 simga_cut_n[81,5]
   + 0.0055987204449463 simga_cut_n[81,6]
   + 0.00762048060562135 simga_cut_n[81,7]
   + 0.00995328079101564 simga_cut_n[81,8]
   + 0.0125971210011292 simga_cut_n[81,9] + 0.00011907 simga_cut_n[82,1]
   + 0.00047628 simga_cut_n[82,2] + 0.00107163 simga_cut_n[82,3]
   + 0.00190512 simga_cut_n[82,4] + 0.00297675 simga_cut_n[82,5]
   + 0.00428652 simga_cut_n[82,6] + 0.00583443 simga_cut_n[82,7]
   + 0.00762047988464356 simga_cut_n[82,8]
   + 0.009644669854002 simga_cut_n[82,9] + 0.00011907 simga_cut_n[83,1]
   + 0.00047628 simga_cut_n[83,2] + 0.00107163 simga_cut_n[83,3]
   + 0.00190512 simga_cut_n[83,4] + 0.00297675 simga_cut_n[83,5]
   + 0.00428652 simga_cut_n[83,6] + 0.00583443 simga_cut_n[83,7]
   + 0.00762047988464356 simga_cut_n[83,8]
   + 0.009644669854002 simga_cut_n[83,9] + 0.00034347 simga_cut_n[84,1]
   + 0.00137388013469696 simga_cut_n[84,2]
   + 0.00309123030306817 simga_cut_n[84,3]
   + 0.00549552053878785 simga_cut_n[84,4]
   + 0.00858675084185602 simga_cut_n[84,5]
   + 0.0123649212122727 simga_cut_n[84,6]
   + 0.0168300316500378 simga_cut_n[84,7]
   + 0.0219820821551514 simga_cut_n[84,8]
   + 0.0278210727276135 simga_cut_n[84,9] + 0.00034347 simga_cut_n[85,1]
   + 0.00137388013469696 simga_cut_n[85,2]
   + 0.00309123030306817 simga_cut_n[85,3]
   + 0.00549552053878785 simga_cut_n[85,4]
   + 0.00858675084185602 simga_cut_n[85,5]
   + 0.0123649212122727 simga_cut_n[85,6]
   + 0.0168300316500378 simga_cut_n[85,7]
   + 0.0219820821551514 simga_cut_n[85,8]
   + 0.0278210727276135 simga_cut_n[85,9] + 0.00034347 simga_cut_n[86,1]
   + 0.00137388013469696 simga_cut_n[86,2]
   + 0.00309123030306817 simga_cut_n[86,3]
   + 0.00549552053878785 simga_cut_n[86,4]
   + 0.00858675084185602 simga_cut_n[86,5]
   + 0.0123649212122727 simga_cut_n[86,6]
   + 0.0168300316500378 simga_cut_n[86,7]
   + 0.0219820821551514 simga_cut_n[86,8]
   + 0.0278210727276135 simga_cut_n[86,9] + 0.00034347 simga_cut_n[87,1]
   + 0.00137388013469696 simga_cut_n[87,2]
   + 0.00309123030306817 simga_cut_n[87,3]
   + 0.00549552053878785 simga_cut_n[87,4]
   + 0.00858675084185602 simga_cut_n[87,5]
   + 0.0123649212122727 simga_cut_n[87,6]
   + 0.0168300316500378 simga_cut_n[87,7]
   + 0.0219820821551514 simga_cut_n[87,8]
   + 0.0278210727276135 simga_cut_n[87,9] + 0.000588 simga_cut_n[88,1]
   + 0.002352 simga_cut_n[88,2] + 0.00529199981975555 simga_cut_n[88,3]
   + 0.00940799967956543 simga_cut_n[88,4]
   + 0.014699999499321 simga_cut_n[88,5]
   + 0.0211679992790222 simga_cut_n[88,6]
   + 0.0288119990186691 simga_cut_n[88,7]
   + 0.0376319987182617 simga_cut_n[88,8]
   + 0.0476279983778 simga_cut_n[88,9] + 0.00052272 simga_cut_n[89,1]
   + 0.00209088016616822 simga_cut_n[89,2]
   + 0.00470448037387849 simga_cut_n[89,3]
   + 0.00836352066467287 simga_cut_n[89,4]
   + 0.0130680010385514 simga_cut_n[89,5]
   + 0.0188179214955139 simga_cut_n[89,6]
   + 0.0256132820355606 simga_cut_n[89,7]
   + 0.0334540826586915 simga_cut_n[89,8]
   + 0.0423403233649064 simga_cut_n[89,9] + 0.000588 simga_cut_n[90,1]
   + 0.002352 simga_cut_n[90,2] + 0.00529199981975555 simga_cut_n[90,3]
   + 0.00940799967956543 simga_cut_n[90,4]
   + 0.014699999499321 simga_cut_n[90,5]
   + 0.0211679992790222 simga_cut_n[90,6]
   + 0.0288119990186691 simga_cut_n[90,7]
   + 0.0376319987182617 simga_cut_n[90,8]
   + 0.0476279983778 simga_cut_n[90,9] + 0.000588 simga_cut_n[91,1]
   + 0.002352 simga_cut_n[91,2] + 0.00529199981975555 simga_cut_n[91,3]
   + 0.00940799967956543 simga_cut_n[91,4]
   + 0.014699999499321 simga_cut_n[91,5]
   + 0.0211679992790222 simga_cut_n[91,6]
   + 0.0288119990186691 simga_cut_n[91,7]
   + 0.0376319987182617 simga_cut_n[91,8]
   + 0.0476279983778 simga_cut_n[91,9] + 0.00088752 simga_cut_n[92,1]
   + 0.00355008011810303 simga_cut_n[92,2]
   + 0.00798768026573181 simga_cut_n[92,3]
   + 0.0142003204724121 simga_cut_n[92,4]
   + 0.0221880007381439 simga_cut_n[92,5]
   + 0.0319507210629272 simga_cut_n[92,6]
   + 0.0434884814467621 simga_cut_n[92,7]
   + 0.0568012818896485 simga_cut_n[92,8]
   + 0.0718891223915863 simga_cut_n[92,9] + 0.00083667 simga_cut_n[93,1]
   + 0.0033466798279953 simga_cut_n[93,2]
   + 0.00753002961298943 simga_cut_n[93,3]
   + 0.0133867193119812 simga_cut_n[93,4]
   + 0.0209167489249706 simga_cut_n[93,5]
   + 0.0301201184519577 simga_cut_n[93,6]
   + 0.0409968278929425 simga_cut_n[93,7]
   + 0.0535468772479248 simga_cut_n[93,8]
   + 0.0677702665169049 simga_cut_n[93,9] + 0.00088752 simga_cut_n[94,1]
   + 0.00355008011810303 simga_cut_n[94,2]
   + 0.00798768026573181 simga_cut_n[94,3]
   + 0.0142003204724121 simga_cut_n[94,4]
   + 0.0221880007381439 simga_cut_n[94,5]
   + 0.0319507210629272 simga_cut_n[94,6]
   + 0.0434884814467621 simga_cut_n[94,7]
   + 0.0568012818896485 simga_cut_n[94,8]
   + 0.0718891223915863 simga_cut_n[94,9] + 0.00088752 simga_cut_n[95,1]
   + 0.00355008011810303 simga_cut_n[95,2]
   + 0.00798768026573181 simga_cut_n[95,3]
   + 0.0142003204724121 simga_cut_n[95,4]
   + 0.0221880007381439 simga_cut_n[95,5]
   + 0.0319507210629272 simga_cut_n[95,6]
   + 0.0434884814467621 simga_cut_n[95,7]
   + 0.0568012818896485 simga_cut_n[95,8]
   + 0.0718891223915863 simga_cut_n[95,9]
   + 0.007537014423043 simga_add_n[35,1]
   + 0.030148057692172 simga_add_n[35,2]
   + 0.067833129807387 simga_add_n[35,3]
   + 0.120592230768688 simga_add_n[35,4]
   + 0.188425360576075 simga_add_n[35,5]
   + 0.271332519229548 simga_add_n[35,6]
   + 0.3693137067291071 simga_add_n[35,7]
   + 0.4823689230747521 simga_add_n[35,8]
   + 0.6104981682664832 simga_add_n[35,9]
   + 0.0930142096471507 simga_add_n[36,1]
   + 0.3720568385886029 simga_add_n[36,2]
   + 0.8371278868243563 simga_add_n[36,3]
   + 1.488227354354412 simga_add_n[36,4]
   + 2.325355241178768 simga_add_n[36,5]
   + 3.348511547297425 simga_add_n[36,6]
   + 4.557696272710385 simga_add_n[36,7]
   + 5.952909417417646 simga_add_n[36,8]
   + 7.534150981419208 simga_add_n[36,9]
   + 0.2673738814450917 simga_add_n[37,1]
   + 1.069495525780367 simga_add_n[37,2]
   + 2.406364933005825 simga_add_n[37,3]
   + 4.277982103121467 simga_add_n[37,4]
   + 6.684347036127292 simga_add_n[37,5]
   + 9.6254597320233 simga_add_n[37,6]
   + 13.10132019080949 simga_add_n[37,7]
   + 17.11192841248587 simga_add_n[37,8]
   + 21.65728439705243 simga_add_n[37,9]
   + 0.415216173929031 simga_add_n[38,1]
   + 1.660864695716124 simga_add_n[38,2]
   + 3.736945565361277 simga_add_n[38,3]
   + 6.643458782864496 simga_add_n[38,4]
   + 10.38040434822577 simga_add_n[38,5]
   + 14.94778226144511 simga_add_n[38,6]
   + 20.34559252252251 simga_add_n[38,7]
   + 26.57383513145798 simga_add_n[38,8]
   + 33.63251008825151 simga_add_n[38,9]
   + 0.6457353511383888 simga_add_n[39,1]
   + 2.582941404553555 simga_add_n[39,2]
   + 5.8116181602455 simga_add_n[39,3]
   + 10.33176561821422 simga_add_n[39,4]
   + 16.14338377845972 simga_add_n[39,5]
   + 23.246472640982 simga_add_n[39,6]
   + 31.64103220578105 simga_add_n[39,7]
   + 41.32706247285688 simga_add_n[39,8]
   + 52.30456344220949 simga_add_n[39,9]
   + 0.7643673612861366 simga_add_n[40,1]
   + 3.057469445144546 simga_add_n[40,2]
   + 6.879306251575231 simga_add_n[40,3]
   + 12.22987778057819 simga_add_n[40,4]
   + 19.10918403215342 simga_add_n[40,5]
   + 27.51722500630092 simga_add_n[40,6]
   + 37.4540007030207 simga_add_n[40,7]
   + 48.91951112231274 simga_add_n[40,8]
   + 61.91375626417707 simga_add_n[40,9]
   + 0.8625478357986086 simga_add_n[41,1]
   + 3.450191343194434 simga_add_n[41,2]
   + 7.762930522187477 simga_add_n[41,3]
   + 13.80076537277774 simga_add_n[41,4]
   + 21.56369589496522 simga_add_n[41,5]
   + 31.05172208874991 simga_add_n[41,6]
   + 42.26484395413183 simga_add_n[41,7]
   + 55.20306149111095 simga_add_n[41,8]
   + 69.86637469968728 simga_add_n[41,9]
   + 0.878084888341132 simga_add_n[42,1]
   + 3.512339553364528 simga_add_n[42,2]
   + 7.902763995070188 simga_add_n[42,3]
   + 14.04935821345811 simga_add_n[42,4]
   + 21.9521222085283 simga_add_n[42,5]
   + 31.61105598028075 simga_add_n[42,6]
   + 43.02615952871546 simga_add_n[42,7]
   + 56.19743285383245 simga_add_n[42,8]
   + 71.12487595563168 simga_add_n[42,9]
   + 1.577688000119139 simga_add_n[43,1]
   + 6.310752000476556 simga_add_n[43,2]
   + 14.19919200107225 simga_add_n[43,3]
   + 25.24300800190622 simga_add_n[43,4]
   + 39.44220000297847 simga_add_n[43,5]
   + 56.79676800428899 simga_add_n[43,6]
   + 77.3067120058378 simga_add_n[43,7]
   + 100.9720320076249 simga_add_n[43,8]
   + 127.7927280096503 simga_add_n[43,9]
   + 1.604706051742499 simga_add_n[44,1]
   + 6.418824206969996 simga_add_n[44,2]
   + 14.44235446568249 simga_add_n[44,3]
   + 25.67529682787998 simga_add_n[44,4]
   + 40.11765129356247 simga_add_n[44,5]
   + 57.76941786272996 simga_add_n[44,6]
   + 78.63059653538244 simga_add_n[44,7]
   + 102.7011873115199 simga_add_n[44,8]
   + 129.9811901911424 simga_add_n[44,9]
   + 1.706665510032677 simga_add_n[45,1]
   + 6.826662040130708 simga_add_n[45,2]
   + 15.35998959029409 simga_add_n[45,3]
   + 27.30664816052283 simga_add_n[45,4]
   + 42.66663775081692 simga_add_n[45,5]
   + 61.43995836117637 simga_add_n[45,6]
   + 83.62660999160119 simga_add_n[45,7]
   + 109.2265926420913 simga_add_n[45,8]
   + 138.2399063126468 simga_add_n[45,9]
   + 1.255926256976304 simga_add_n[46,1]
   + 5.023705027905215 simga_add_n[46,2]
   + 11.30333631278673 simga_add_n[46,3]
   + 20.09482011162086 simga_add_n[46,4]
   + 31.3981564244076 simga_add_n[46,5]
   + 45.21334525114693 simga_add_n[46,6]
   + 61.54038659183886 simga_add_n[46,7]
   + 80.37928044648343 simga_add_n[46,8]
   + 101.7300268150806 simga_add_n[46,9]
   + 1.294342050307676 simga_add_n[47,1]
   + 5.177368201230703 simga_add_n[47,2]
   + 11.64907845276908 simga_add_n[47,3]
   + 20.70947280492281 simga_add_n[47,4]
   + 32.3585512576919 simga_add_n[47,5]
   + 46.59631381107634 simga_add_n[47,6]
   + 63.42276046507611 simga_add_n[47,7]
   + 82.83789121969124 simga_add_n[47,8]
   + 104.8417060749217 simga_add_n[47,9]
   + 1.249961654835133 simga_add_n[48,1]
   + 4.999846619340533 simga_add_n[48,2]
   + 11.2496548935162 simga_add_n[48,3]
   + 19.99938647736213 simga_add_n[48,4]
   + 31.24904137087833 simga_add_n[48,5]
   + 44.9986195740648 simga_add_n[48,6]
   + 61.24812108692154 simga_add_n[48,7]
   + 79.99754590944853 simga_add_n[48,8]
   + 101.2468940416458 simga_add_n[48,9]
   + 1.402563036531077 simga_add_n[49,1]
   + 5.610252146124309 simga_add_n[49,2]
   + 12.62306732877969 simga_add_n[49,3]
   + 22.44100858449724 simga_add_n[49,4]
   + 35.06407591327693 simga_add_n[49,5]
   + 50.49226931511878 simga_add_n[49,6]
   + 68.72558879002278 simga_add_n[49,7]
   + 89.76403433798895 simga_add_n[49,8]
   + 113.6076059590173 simga_add_n[49,9]
   + 1.260883034883157 simga_add_n[50,1]
   + 5.04353213953263 simga_add_n[50,2]
   + 11.34794731394842 simga_add_n[50,3]
   + 20.17412855813052 simga_add_n[50,4]
   + 31.52207587207894 simga_add_n[50,5]
   + 45.39178925579368 simga_add_n[50,6]
   + 61.78326870927471 simga_add_n[50,7]
   + 80.69651423252208 simga_add_n[50,8]
   + 102.1315258255358 simga_add_n[50,9]
   + 1.298371083850548 simga_add_n[51,1]
   + 5.193484335402192 simga_add_n[51,2]
   + 11.68533975465493 simga_add_n[51,3]
   + 20.77393734160877 simga_add_n[51,4]
   + 32.4592770962637 simga_add_n[51,5]
   + 46.74135901861973 simga_add_n[51,6]
   + 63.62018310867685 simga_add_n[51,7]
   + 83.09574936643507 simga_add_n[51,8]
   + 105.1680577918944 simga_add_n[51,9]
   + 1.242803865714872 simga_add_n[52,1]
   + 4.971215462859488 simga_add_n[52,2]
   + 11.18523479143385 simga_add_n[52,3]
   + 19.88486185143795 simga_add_n[52,4]
   + 31.0700966428718 simga_add_n[52,5]
   + 44.74093916573539 simga_add_n[52,6]
   + 60.89738942002872 simga_add_n[52,7]
   + 79.5394474057518 simga_add_n[52,8]
   + 100.6671131229046 simga_add_n[52,9]
   + 1.21928460972131 simga_add_n[53,1]
   + 4.87713843888524 simga_add_n[53,2]
   + 10.97356148749179 simga_add_n[53,3]
   + 19.50855375554096 simga_add_n[53,4]
   + 30.48211524303274 simga_add_n[53,5]
   + 43.89424594996716 simga_add_n[53,6]
   + 59.74494587634419 simga_add_n[53,7]
   + 78.03421502216383 simga_add_n[53,8]
   + 98.76205338742609 simga_add_n[53,9]
   + 1.266547601741711 simga_add_n[54,1]
   + 5.066190406966845 simga_add_n[54,2]
   + 11.3989284156754 simga_add_n[54,3]
   + 20.26476162786738 simga_add_n[54,4]
   + 31.66369004354278 simga_add_n[54,5]
   + 45.5957136627016 simga_add_n[54,6]
   + 62.06083248534384 simga_add_n[54,7]
   + 81.05904651146952 simga_add_n[54,8]
   + 102.5903557410786 simga_add_n[54,9]
   + 1.310997001949141 simga_add_n[55,1]
   + 5.243988007796564 simga_add_n[55,2]
   + 11.79897301754227 simga_add_n[55,3]
   + 20.97595203118626 simga_add_n[55,4]
   + 32.77492504872853 simga_add_n[55,5]
   + 47.19589207016909 simga_add_n[55,6]
   + 64.23885309550791 simga_add_n[55,7]
   + 83.90380812474503 simga_add_n[55,8]
   + 106.1907571578805 simga_add_n[55,9] + 1.451070576353 simga_add_n[56,1]
   + 5.804282305412001 simga_add_n[56,2]
   + 13.059635187177 simga_add_n[56,3] + 23.217129221648 simga_add_n[56,4]
   + 36.27676440882501 simga_add_n[56,5]
   + 52.23854074870801 simga_add_n[56,6]
   + 71.10245824129701 simga_add_n[56,7]
   + 92.86851688659202 simga_add_n[56,8]
   + 117.536716684593 simga_add_n[56,9]
   + 1.532884706027474 simga_add_n[57,1]
   + 6.131538824109894 simga_add_n[57,2]
   + 13.79596235424726 simga_add_n[57,3]
   + 24.52615529643958 simga_add_n[57,4]
   + 38.32211765068683 simga_add_n[57,5]
   + 55.18384941698904 simga_add_n[57,6]
   + 75.11135059534618 simga_add_n[57,7]
   + 98.1046211857583 simga_add_n[57,8]
   + 124.1636611882253 simga_add_n[57,9]
   + 1.481839487935414 simga_add_n[58,1]
   + 5.927357951741656 simga_add_n[58,2]
   + 13.33655539141873 simga_add_n[58,3]
   + 23.70943180696662 simga_add_n[58,4]
   + 37.04598719838535 simga_add_n[58,5]
   + 53.34622156567492 simga_add_n[58,6]
   + 72.61013490883531 simga_add_n[58,7]
   + 94.83772722786649 simga_add_n[58,8]
   + 120.0289985227686 simga_add_n[58,9]
   + 1.709278311430384 simga_add_n[59,1]
   + 6.837113245721534 simga_add_n[59,2]
   + 15.38350480287346 simga_add_n[59,3]
   + 27.34845298288614 simga_add_n[59,4]
   + 42.7319577857596 simga_add_n[59,5]
   + 61.53401921149383 simga_add_n[59,6]
   + 83.75463726008883 simga_add_n[59,7]
   + 109.3938119315445 simga_add_n[59,8]
   + 138.4515432258611 simga_add_n[59,9]
   + 1.4740063577148 simga_add_n[60,1]
   + 5.896025430859201 simga_add_n[60,2]
   + 13.26605721943321 simga_add_n[60,3]
   + 23.58410172343681 simga_add_n[60,4]
   + 36.85015894287001 simga_add_n[60,5]
   + 53.06422887773283 simga_add_n[60,6]
   + 72.22631152802522 simga_add_n[60,7]
   + 94.33640689374722 simga_add_n[60,8]
   + 119.3945149748988 simga_add_n[60,9]
   + 1.277524484869077 simga_add_n[61,1]
   + 5.110097939476309 simga_add_n[61,2]
   + 11.4977203638217 simga_add_n[61,3]
   + 20.44039175790524 simga_add_n[61,4]
   + 31.93811212172693 simga_add_n[61,5]
   + 45.99088145528678 simga_add_n[61,6]
   + 62.59869975858477 simga_add_n[61,7]
   + 81.76156703162094 simga_add_n[61,8]
   + 103.4794832743952 simga_add_n[61,9]
   + 1.418481074029165 simga_add_n[62,1]
   + 5.673924296116659 simga_add_n[62,2]
   + 12.76632966626248 simga_add_n[62,3]
   + 22.69569718446663 simga_add_n[62,4]
   + 35.46202685072911 simga_add_n[62,5]
   + 51.06531866504992 simga_add_n[62,6]
   + 69.50557262742907 simga_add_n[62,7]
   + 90.78278873786654 simga_add_n[62,8]
   + 114.8969669963623 simga_add_n[62,9]
   + 1.08278337748943 simga_add_n[63,1]
   + 4.331133509957718 simga_add_n[63,2]
   + 9.745050397404867 simga_add_n[63,3]
   + 17.32453403983087 simga_add_n[63,4]
   + 27.06958443723574 simga_add_n[63,5]
   + 38.98020158961947 simga_add_n[63,6]
   + 53.05638549698203 simga_add_n[63,7]
   + 69.29813615932349 simga_add_n[63,8]
   + 87.70545357664383 simga_add_n[63,9]
   + 0.5868243681303741 simga_add_n[64,1]
   + 2.347297472521496 simga_add_n[64,2]
   + 5.281419313173367 simga_add_n[64,3]
   + 9.389189890085985 simga_add_n[64,4]
   + 14.67060920325935 simga_add_n[64,5]
   + 21.12567725269347 simga_add_n[64,6]
   + 28.75439403838834 simga_add_n[64,7]
   + 37.55675956034394 simga_add_n[64,8]
   + 47.53277381856031 simga_add_n[64,9]
   + 0.3958833293505081 simga_add_n[65,1]
   + 1.583533317402032 simga_add_n[65,2]
   + 3.562949964154573 simga_add_n[65,3]
   + 6.334133269608129 simga_add_n[65,4]
   + 9.8970832337627 simga_add_n[65,5]
   + 14.25179985661829 simga_add_n[65,6]
   + 19.39828313817489 simga_add_n[65,7]
   + 25.33653307843252 simga_add_n[65,8]
   + 32.06654967739114 simga_add_n[65,9]
   + 0.00526832528784218 simga_add_n[66,1]
   + 0.0210733011513687 simga_add_n[66,2]
   + 0.0474149275905796 simga_add_n[66,3]
   + 0.0842932046054749 simga_add_n[66,4]
   + 0.1317081321960545 simga_add_n[66,5]
   + 0.1896597103623184 simga_add_n[66,6]
   + 0.2581479391042668 simga_add_n[66,7]
   + 0.3371728184218995 simga_add_n[66,8]
   + 0.4267343483152164 simga_add_n[66,9]
   + 0.00231341985614794 simga_add_n[67,1]
   + 0.00925367942459175 simga_add_n[67,2]
   + 0.0208207787053314 simga_add_n[67,3]
   + 0.037014717698367 simga_add_n[67,4]
   + 0.0578354964036984 simga_add_n[67,5]
   + 0.0832831148213257 simga_add_n[67,6]
   + 0.1133575729512489 simga_add_n[67,7]
   + 0.1480588707934679 simga_add_n[67,8]
   + 0.1873870083479829 simga_add_n[67,9]
Subject To
 c_cost_fuel_PWL[0]: x_cost_fuel_PWL[0] = 99.43502934598924
 c_cost_fuel_PWL[1]: x_cost_fuel_PWL[1] = 98.37107258939744
 c_cost_fuel_PWL[2]: x_cost_fuel_PWL[2] = 99.9946815981865
 c_cost_fuel_PWL[3]: x_cost_fuel_PWL[3] = 99.22593159818649
 c_cost_fuel_PWL[4]: x_cost_fuel_PWL[4] = 99.43502934598924
 c_cost_fuel_PWL[5]: x_cost_fuel_PWL[5] = 100.616460770607
 c_cost_fuel_PWL[6]: x_cost_fuel_PWL[6] = 98.13185551548004
 c_cost_fuel_PWL[7]: x_cost_fuel_PWL[7] = 100.3212540140152
 c_cost_fuel_PWL[8]: x_cost_fuel_PWL[8] = 99.37414875888825
 c_cost_fuel_PWL[9]: x_cost_fuel_PWL[9] = 100.0506525125504
 c_cost_fuel_PWL[10]: x_cost_fuel_PWL[10] = 100.555580183506
 c_cost_fuel_PWL[11]: x_cost_fuel_PWL[11] = 100.0512756791115
 c_cost_fuel_PWL[12]: x_cost_fuel_PWL[12] = 99.54697117471696
 c_cost_fuel_PWL[13]: x_cost_fuel_PWL[13] = 101.170580183506
 c_cost_fuel_PWL[14]: x_cost_fuel_PWL[14] = 98.98731892251969
 c_cost_fuel_PWL[15]: x_cost_fuel_PWL[15] = 101.1404608573914
 c_cost_fuel_PWL[16]: x_cost_fuel_PWL[16] = 98.10235935592652
 c_cost_fuel_PWL[17]: x_cost_fuel_PWL[17] = 99.31391010665894
 c_cost_fuel_PWL[18]: x_cost_fuel_PWL[18] = 98.80960560226441
 c_cost_fuel_PWL[19]: x_cost_fuel_PWL[19] = 98.48365635299683
 c_cost_fuel_PWL[20]: x_cost_fuel_PWL[20] = 98.98242801809312
 c_cost_fuel_PWL[21]: x_cost_fuel_PWL[21] = 98.15833027029038
 c_cost_fuel_PWL[22]: x_cost_fuel_PWL[22] = 98.51502201223374
 c_cost_fuel_PWL[23]: x_cost_fuel_PWL[23] = 90.66834279277833
 c_cost_fuel_PWL[24]: x_cost_fuel_PWL[24] = 97.59867801809311
 c_cost_fuel_PWL[25]: x_cost_fuel_PWL[25] = 98.86619968318939
 c_cost_fuel_PWL[26]: x_cost_fuel_PWL[26] = 77.009
 c_cost_fuel_PWL[27]: x_cost_fuel_PWL[27] = 92.09098385715485
 c_cost_fuel_PWL[28]: x_cost_fuel_PWL[28] = 77.009
 c_cost_fuel_PWL[29]: x_cost_fuel_PWL[29] = 78.90049145126343
 c_cost_fuel_PWL[30]: x_cost_fuel_PWL[30] = 84.7648776473999
 c_cost_fuel_PWL[31]: x_cost_fuel_PWL[31] = 63.89200228118897
 c_cost_fuel_PWL[32]: x_cost_fuel_PWL[32] = 45.9844207611084
 c_cost_fuel_PWL[33]: x_cost_fuel_PWL[33] = 34.87641905212402
 c_cost_fuel_PWL[34]: x_cost_fuel_PWL[34] = 27.09402464294434
 c_cost_fuel_PWL[35]: x_cost_fuel_PWL[35] = 24.809
 c_cost_fuel_PWL[36]: x_cost_fuel_PWL[36] = 24.809
 c_cost_fuel_PWL[37]: x_cost_fuel_PWL[37] = 24.809
 c_cost_fuel_PWL[38]: x_cost_fuel_PWL[38] = 24.809
 c_cost_fuel_PWL[39]: x_cost_fuel_PWL[39] = 24.809
 c_cost_fuel_PWL[40]: x_cost_fuel_PWL[40] = 24.809
 c_cost_fuel_PWL[41]: x_cost_fuel_PWL[41] = 24.809
 c_cost_fuel_PWL[42]: x_cost_fuel_PWL[42] = 24.809
 c_cost_fuel_PWL[43]: x_cost_fuel_PWL[43] = 24.809
 c_cost_fuel_PWL[44]: x_cost_fuel_PWL[44] = 24.809
 c_cost_fuel_PWL[45]: x_cost_fuel_PWL[45] = 24.809
 c_cost_fuel_PWL[46]: x_cost_fuel_PWL[46] = 24.809
 c_cost_fuel_PWL[47]: x_cost_fuel_PWL[47] = 24.809
 c_cost_fuel_PWL[48]: x_cost_fuel_PWL[48] = 24.809
 c_cost_fuel_PWL[49]: x_cost_fuel_PWL[49] = 24.809
 c_cost_fuel_PWL[50]: x_cost_fuel_PWL[50] = 24.809
 c_cost_fuel_PWL[51]: x_cost_fuel_PWL[51] = 24.809
 c_cost_fuel_PWL[52]: x_cost_fuel_PWL[52] = 24.809
 c_cost_fuel_PWL[53]: x_cost_fuel_PWL[53] = 24.809
 c_cost_fuel_PWL[54]: x_cost_fuel_PWL[54] = 24.809
 c_cost_fuel_PWL[55]: x_cost_fuel_PWL[55] = 24.809
 c_cost_fuel_PWL[56]: x_cost_fuel_PWL[56] = 24.809
 c_cost_fuel_PWL[57]: x_cost_fuel_PWL[57] = 24.809
 c_cost_fuel_PWL[58]: x_cost_fuel_PWL[58] = 24.809
 c_cost_fuel_PWL[59]: x_cost_fuel_PWL[59] = 24.809
 c_cost_fuel_PWL[60]: x_cost_fuel_PWL[60] = 24.809
 c_cost_fuel_PWL[61]: x_cost_fuel_PWL[61] = 24.809
 c_cost_fuel_PWL[62]: x_cost_fuel_PWL[62] = 24.809
 c_cost_fuel_PWL[63]: x_cost_fuel_PWL[63] = 24.809
 c_cost_fuel_PWL[64]: x_cost_fuel_PWL[64] = 24.809
 c_cost_fuel_PWL[65]: x_cost_fuel_PWL[65] = 24.809
 c_cost_fuel_PWL[66]: x_cost_fuel_PWL[66] = 24.809
 c_cost_fuel_PWL[67]: x_cost_fuel_PWL[67] = 24.809
 c_cost_fuel_PWL[68]: x_cost_fuel_PWL[68] = 46.32567213439941
 c_cost_fuel_PWL[69]: x_cost_fuel_PWL[69] = 42.94075868988037
 c_cost_fuel_PWL[70]: x_cost_fuel_PWL[70] = 47.309
 c_cost_fuel_PWL[71]: x_cost_fuel_PWL[71] = 67.21973978424072
 c_cost_fuel_PWL[72]: x_cost_fuel_PWL[72] = 65.5339009513855
 c_cost_fuel_PWL[73]: x_cost_fuel_PWL[73] = 67.83111795806884
 c_cost_fuel_PWL[74]: x_cost_fuel_PWL[74] = 77.009
 c_cost_fuel_PWL[75]: x_cost_fuel_PWL[75] = 77.009
 c_cost_fuel_PWL[76]: x_cost_fuel_PWL[76] = 77.009
 c_cost_fuel_PWL[77]: x_cost_fuel_PWL[77] = 77.009
 c_cost_fuel_PWL[78]: x_cost_fuel_PWL[78] = 77.009
 c_cost_fuel_PWL[79]: x_cost_fuel_PWL[79] = 77.009
 c_cost_fuel_PWL[80]: x_cost_fuel_PWL[80] = 77.009
 c_cost_fuel_PWL[81]: x_cost_fuel_PWL[81] = 77.009
 c_cost_fuel_PWL[82]: x_cost_fuel_PWL[82] = 77.009
 c_cost_fuel_PWL[83]: x_cost_fuel_PWL[83] = 77.009
 c_cost_fuel_PWL[84]: x_cost_fuel_PWL[84] = 77.009
 c_cost_fuel_PWL[85]: x_cost_fuel_PWL[85] = 77.009
 c_cost_fuel_PWL[86]: x_cost_fuel_PWL[86] = 77.009
 c_cost_fuel_PWL[87]: x_cost_fuel_PWL[87] = 77.009
 c_cost_fuel_PWL[88]: x_cost_fuel_PWL[88] = 77.009
 c_cost_fuel_PWL[89]: x_cost_fuel_PWL[89] = 77.009
 c_cost_fuel_PWL[90]: x_cost_fuel_PWL[90] = 77.009
 c_cost_fuel_PWL[91]: x_cost_fuel_PWL[91] = 77.009
 c_cost_fuel_PWL[92]: x_cost_fuel_PWL[92] = 77.009
 c_cost_fuel_PWL[93]: x_cost_fuel_PWL[93] = 77.009
 c_cost_fuel_PWL[94]: x_cost_fuel_PWL[94] = 77.009
 c_cost_fuel_PWL[95]: x_cost_fuel_PWL[95] = 50.46348799022654
 c_cost_fuel_res[0]: x_cost_res[0] = 0
 c_cost_fuel_res[1]: x_cost_res[1] = 0
 c_cost_fuel_res[2]: x_cost_res[2] = 0
 c_cost_fuel_res[3]: x_cost_res[3] = 0
 c_cost_fuel_res[4]: x_cost_res[4] = 0
 c_cost_fuel_res[5]: x_cost_res[5] = 0
 c_cost_fuel_res[6]: x_cost_res[6] = 0
 c_cost_fuel_res[7]: x_cost_res[7] = 0
 c_cost_fuel_res[8]: x_cost_res[8] = 0
 c_cost_fuel_res[9]: x_cost_res[9] = 0
 c_cost_fuel_res[10]: x_cost_res[10] = 0
 c_cost_fuel_res[11]: x_cost_res[11] = 0
 c_cost_fuel_res[12]: x_cost_res[12] = 0
 c_cost_fuel_res[13]: x_cost_res[13] = 0
 c_cost_fuel_res[14]: x_cost_res[14] = 0
 c_cost_fuel_res[15]: x_cost_res[15] = 0
 c_cost_fuel_res[16]: x_cost_res[16] = 0
 c_cost_fuel_res[17]: x_cost_res[17] = 0
 c_cost_fuel_res[18]: x_cost_res[18] = 0
 c_cost_fuel_res[19]: x_cost_res[19] = 0
 c_cost_fuel_res[20]: x_cost_res[20] = 0
 c_cost_fuel_res[21]: x_cost_res[21] = 0
 c_cost_fuel_res[22]: x_cost_res[22] = 0
 c_cost_fuel_res[23]: x_cost_res[23] = 0
 c_cost_fuel_res[24]: x_cost_res[24] = 0
 c_cost_fuel_res[25]: x_cost_res[25] = 0
 c_cost_fuel_res[26]: x_cost_res[26] = 0
 c_cost_fuel_res[27]: x_cost_res[27] = 0
 c_cost_fuel_res[28]: x_cost_res[28] = 0
 c_cost_fuel_res[29]: x_cost_res[29] = 0
 c_cost_fuel_res[30]: x_cost_res[30] = 0
 c_cost_fuel_res[31]: x_cost_res[31] = 0
 c_cost_fuel_res[32]: x_cost_res[32] = 0
 c_cost_fuel_res[33]: x_cost_res[33] = 0
 c_cost_fuel_res[34]: x_cost_res[34] = 0
 c_cost_fuel_res[35]: x_cost_res[35] = 0
 c_cost_fuel_res[36]: x_cost_res[36] = 0
 c_cost_fuel_res[37]: x_cost_res[37] = 0
 c_cost_fuel_res[38]: x_cost_res[38] = 0
 c_cost_fuel_res[39]: x_cost_res[39] = 0
 c_cost_fuel_res[40]: x_cost_res[40] = 0
 c_cost_fuel_res[41]: x_cost_res[41] = 0
 c_cost_fuel_res[42]: x_cost_res[42] = 0
 c_cost_fuel_res[43]: x_cost_res[43] = 0
 c_cost_fuel_res[44]: x_cost_res[44] = 0
 c_cost_fuel_res[45]: x_cost_res[45] = 0
 c_cost_fuel_res[46]: x_cost_res[46] = 0
 c_cost_fuel_res[47]: x_cost_res[47] = 0
 c_cost_fuel_res[48]: x_cost_res[48] = 0
 c_cost_fuel_res[49]: x_cost_res[49] = 0
 c_cost_fuel_res[50]: x_cost_res[50] = 0
 c_cost_fuel_res[51]: x_cost_res[51] = 0
 c_cost_fuel_res[52]: x_cost_res[52] = 0
 c_cost_fuel_res[53]: x_cost_res[53] = 0
 c_cost_fuel_res[54]: x_cost_res[54] = 0
 c_cost_fuel_res[55]: x_cost_res[55] = 0
 c_cost_fuel_res[56]: x_cost_res[56] = 0
 c_cost_fuel_res[57]: x_cost_res[57] = 0
 c_cost_fuel_res[58]: x_cost_res[58] = 0
 c_cost_fuel_res[59]: x_cost_res[59] = 0
 c_cost_fuel_res[60]: x_cost_res[60] = 0
 c_cost_fuel_res[61]: x_cost_res[61] = 0
 c_cost_fuel_res[62]: x_cost_res[62] = 0
 c_cost_fuel_res[63]: x_cost_res[63] = 0
 c_cost_fuel_res[64]: x_cost_res[64] = 0
 c_cost_fuel_res[65]: x_cost_res[65] = 0
 c_cost_fuel_res[66]: x_cost_res[66] = 0
 c_cost_fuel_res[67]: x_cost_res[67] = 0
 c_cost_fuel_res[68]: x_cost_res[68] = 0
 c_cost_fuel_res[69]: x_cost_res[69] = 0
 c_cost_fuel_res[70]: x_cost_res[70] = 0
 c_cost_fuel_res[71]: x_cost_res[71] = 0
 c_cost_fuel_res[72]: x_cost_res[72] = 0
 c_cost_fuel_res[73]: x_cost_res[73] = 0
 c_cost_fuel_res[74]: x_cost_res[74] = 0
 c_cost_fuel_res[75]: x_cost_res[75] = 0
 c_cost_fuel_res[76]: x_cost_res[76] = 0
 c_cost_fuel_res[77]: x_cost_res[77] = 0
 c_cost_fuel_res[78]: x_cost_res[78] = 0
 c_cost_fuel_res[79]: x_cost_res[79] = 0
 c_cost_fuel_res[80]: x_cost_res[80] = 0
 c_cost_fuel_res[81]: x_cost_res[81] = 0
 c_cost_fuel_res[82]: x_cost_res[82] = 0
 c_cost_fuel_res[83]: x_cost_res[83] = 0
 c_cost_fuel_res[84]: x_cost_res[84] = 0
 c_cost_fuel_res[85]: x_cost_res[85] = 0
 c_cost_fuel_res[86]: x_cost_res[86] = 0
 c_cost_fuel_res[87]: x_cost_res[87] = 0
 c_cost_fuel_res[88]: x_cost_res[88] = 0
 c_cost_fuel_res[89]: x_cost_res[89] = 0
 c_cost_fuel_res[90]: x_cost_res[90] = 0
 c_cost_fuel_res[91]: x_cost_res[91] = 0
 c_cost_fuel_res[92]: x_cost_res[92] = 0
 c_cost_fuel_res[93]: x_cost_res[93] = 0
 c_cost_fuel_res[94]: x_cost_res[94] = 0
 c_cost_fuel_res[95]: x_cost_res[95] = 0
 c_cost_ESS_OM_pre[0]: x_cost_ESS[0] = 0
 c_cost_ESS_OM_pre[1]: x_cost_ESS[1] = 0
 c_cost_ESS_OM_pre[2]: x_cost_ESS[2] = 0
 c_cost_ESS_OM_pre[3]: x_cost_ESS[3] = 0
 c_cost_ESS_OM_pre[4]: x_cost_ESS[4] = 0
 c_cost_ESS_OM_pre[5]: x_cost_ESS[5] = 0
 c_cost_ESS_OM_pre[6]: x_cost_ESS[6] = 0
 c_cost_ESS_OM_pre[7]: x_cost_ESS[7] = 0
 c_cost_ESS_OM_pre[8]: x_cost_ESS[8] = 0
 c_cost_ESS_OM_pre[9]: x_cost_ESS[9] = 0
 c_cost_ESS_OM_pre[10]: x_cost_ESS[10] = 0
 c_cost_ESS_OM_pre[11]: x_cost_ESS[11] = 0
 c_cost_ESS_OM_pre[12]: x_cost_ESS[12] = 0
 c_cost_ESS_OM_pre[13]: x_cost_ESS[13] = 0
 c_cost_ESS_OM_pre[14]: x_cost_ESS[14] = 0
 c_cost_ESS_OM_pre[15]: x_cost_ESS[15] = 0
 c_cost_ESS_OM_pre[16]: x_cost_ESS[16] = 0
 c_cost_ESS_OM_pre[17]: x_cost_ESS[17] = 0
 c_cost_ESS_OM_pre[18]: x_cost_ESS[18] = 0
 c_cost_ESS_OM_pre[19]: x_cost_ESS[19] = 0
 c_cost_ESS_OM_pre[20]: x_cost_ESS[20] = 0
 c_cost_ESS_OM_pre[21]: x_cost_ESS[21] = 0
 c_cost_ESS_OM_pre[22]: x_cost_ESS[22] = 0
 c_cost_ESS_OM_pre[23]: x_cost_ESS[23] = 0.0138088463096614
 c_cost_ESS_OM_pre[24]: x_cost_ESS[24] = 0
 c_cost_ESS_OM_pre[25]: x_cost_ESS[25] = 0
 c_cost_ESS_OM_pre[26]: x_cost_ESS[26] = 0.0325191597938538
 c_cost_ESS_OM_pre[27]: x_cost_ESS[27] = 0
 c_cost_ESS_OM_pre[28]: x_cost_ESS[28] = 0.0183999938964844
 c_cost_ESS_OM_pre[29]: x_cost_ESS[29] = 0
 c_cost_ESS_OM_pre[30]: x_cost_ESS[30] = 0
 c_cost_ESS_OM_pre[31]: x_cost_ESS[31] = 0
 c_cost_ESS_OM_pre[32]: x_cost_ESS[32] = 0
 c_cost_ESS_OM_pre[33]: x_cost_ESS[33] = 0
 c_cost_ESS_OM_pre[34]: x_cost_ESS[34] = 0
 c_cost_ESS_OM_pre[35]: x_cost_ESS[35] = 0
 c_cost_ESS_OM_pre[36]: x_cost_ESS[36] = 0
 c_cost_ESS_OM_pre[37]: x_cost_ESS[37] = 0
 c_cost_ESS_OM_pre[38]: x_cost_ESS[38] = 0.00504125064984808
 c_cost_ESS_OM_pre[39]: x_cost_ESS[39] = 0.00436017768705359
 c_cost_ESS_OM_pre[40]: x_cost_ESS[40] = 0.0130063030482856
 c_cost_ESS_OM_pre[41]: x_cost_ESS[41] = 0.0229404887598702
 c_cost_ESS_OM_pre[42]: x_cost_ESS[42] = 0.0310903802348739
 c_cost_ESS_OM_pre[43]: x_cost_ESS[43] = 0.0161706322639384
 c_cost_ESS_OM_pre[44]: x_cost_ESS[44] = 0.021088769735868
 c_cost_ESS_OM_pre[45]: x_cost_ESS[45] = 0.0279215299307783
 c_cost_ESS_OM_pre[46]: x_cost_ESS[46] = 0.0634427505399441
 c_cost_ESS_OM_pre[47]: x_cost_ESS[47] = 0.0688198639216818
 c_cost_ESS_OM_pre[48]: x_cost_ESS[48] = 0.0741721115072821
 c_cost_ESS_OM_pre[49]: x_cost_ESS[49] = 0.0874193129489238
 c_cost_ESS_OM_pre[50]: x_cost_ESS[50] = 0.0763277841135401
 c_cost_ESS_OM_pre[51]: x_cost_ESS[51] = 0.0773470782100955
 c_cost_ESS_OM_pre[52]: x_cost_ESS[52] = 0.0783552946203955
 c_cost_ESS_OM_pre[53]: x_cost_ESS[53] = 0.0754953829004934
 c_cost_ESS_OM_pre[54]: x_cost_ESS[54] = 0.079765175822111
 c_cost_ESS_OM_pre[55]: x_cost_ESS[55] = 0.0668700220161714
 c_cost_ESS_OM_pre[56]: x_cost_ESS[56] = 0.0267726150284618
 c_cost_ESS_OM_pre[57]: x_cost_ESS[57] = 0.0295564544530375
 c_cost_ESS_OM_pre[58]: x_cost_ESS[58] = 0.0283647677550494
 c_cost_ESS_OM_pre[59]: x_cost_ESS[59] = 0.0395254317199168
 c_cost_ESS_OM_pre[60]: x_cost_ESS[60] = 0.0263915464148844
 c_cost_ESS_OM_pre[61]: x_cost_ESS[61] = 0.0116773470651684
 c_cost_ESS_OM_pre[62]: x_cost_ESS[62] = 0.0267147
 c_cost_ESS_OM_pre[63]: x_cost_ESS[63] = 0.0308876170948921
 c_cost_ESS_OM_pre[64]: x_cost_ESS[64] = 0.00631999418118957
 c_cost_ESS_OM_pre[65]: x_cost_ESS[65] = 0.00730719641714596
 c_cost_ESS_OM_pre[66]: x_cost_ESS[66] = 0.0409119288223717
 c_cost_ESS_OM_pre[67]: x_cost_ESS[67] = 0.0473024960369657
 c_cost_ESS_OM_pre[68]: x_cost_ESS[68] = 0
 c_cost_ESS_OM_pre[69]: x_cost_ESS[69] = 0
 c_cost_ESS_OM_pre[70]: x_cost_ESS[70] = 0.0458374481201172
 c_cost_ESS_OM_pre[71]: x_cost_ESS[71] = 0
 c_cost_ESS_OM_pre[72]: x_cost_ESS[72] = 0
 c_cost_ESS_OM_pre[73]: x_cost_ESS[73] = 0
 c_cost_ESS_OM_pre[74]: x_cost_ESS[74] = 0.00608643531799316
 c_cost_ESS_OM_pre[75]: x_cost_ESS[75] = 0.0236135473251343
 c_cost_ESS_OM_pre[76]: x_cost_ESS[76] = 0.0292932324409485
 c_cost_ESS_OM_pre[77]: x_cost_ESS[77] = 0.0336910157203674
 c_cost_ESS_OM_pre[78]: x_cost_ESS[78] = 0.0328810181617737
 c_cost_ESS_OM_pre[79]: x_cost_ESS[79] = 0.0332610230445862
 c_cost_ESS_OM_pre[80]: x_cost_ESS[80] = 0.0323020448684692
 c_cost_ESS_OM_pre[81]: x_cost_ESS[81] = 0.031862042427063
 c_cost_ESS_OM_pre[82]: x_cost_ESS[82] = 0.0329220399856567
 c_cost_ESS_OM_pre[83]: x_cost_ESS[83] = 0.0315820436477661
 c_cost_ESS_OM_pre[84]: x_cost_ESS[84] = 0.0322030520439148
 c_cost_ESS_OM_pre[85]: x_cost_ESS[85] = 0.0317630496025085
 c_cost_ESS_OM_pre[86]: x_cost_ESS[86] = 0.0316230654716492
 c_cost_ESS_OM_pre[87]: x_cost_ESS[87] = 0.0325330386161804
 c_cost_ESS_OM_pre[88]: x_cost_ESS[88] = 0.0314740543365479
 c_cost_ESS_OM_pre[89]: x_cost_ESS[89] = 0.0323440799713135
 c_cost_ESS_OM_pre[90]: x_cost_ESS[90] = 0.0326740665435791
 c_cost_ESS_OM_pre[91]: x_cost_ESS[91] = 0.0321440677642822
 c_cost_ESS_OM_pre[92]: x_cost_ESS[92] = 0.0337750859260559
 c_cost_ESS_OM_pre[93]: x_cost_ESS[93] = 0.0329650883674622
 c_cost_ESS_OM_pre[94]: x_cost_ESS[94] = 0.0342050786018372
 c_cost_ESS_OM_pre[95]: x_cost_ESS[95] = 0.0879323816947941
 c_cost_PV_cut_PWL[0]: x_cost_cut_PWL[0] = 0
 c_cost_PV_cut_PWL[1]: x_cost_cut_PWL[1] = 0
 c_cost_PV_cut_PWL[2]: x_cost_cut_PWL[2] = 0
 c_cost_PV_cut_PWL[3]: x_cost_cut_PWL[3] = 0
 c_cost_PV_cut_PWL[4]: x_cost_cut_PWL[4] = 0
 c_cost_PV_cut_PWL[5]: x_cost_cut_PWL[5] = 0
 c_cost_PV_cut_PWL[6]: x_cost_cut_PWL[6] = 0
 c_cost_PV_cut_PWL[7]: x_cost_cut_PWL[7] = 0
 c_cost_PV_cut_PWL[8]: x_cost_cut_PWL[8] = 0
 c_cost_PV_cut_PWL[9]: x_cost_cut_PWL[9] = 0
 c_cost_PV_cut_PWL[10]: x_cost_cut_PWL[10] = 0
 c_cost_PV_cut_PWL[11]: x_cost_cut_PWL[11] = 0
 c_cost_PV_cut_PWL[12]: x_cost_cut_PWL[12] = 0
 c_cost_PV_cut_PWL[13]: x_cost_cut_PWL[13] = 0
 c_cost_PV_cut_PWL[14]: x_cost_cut_PWL[14] = 0
 c_cost_PV_cut_PWL[15]: x_cost_cut_PWL[15] = 0
 c_cost_PV_cut_PWL[16]: x_cost_cut_PWL[16] = 0
 c_cost_PV_cut_PWL[17]: x_cost_cut_PWL[17] = 0
 c_cost_PV_cut_PWL[18]: x_cost_cut_PWL[18] = 0
 c_cost_PV_cut_PWL[19]: x_cost_cut_PWL[19] = 0
 c_cost_PV_cut_PWL[20]: x_cost_cut_PWL[20] = 0
 c_cost_PV_cut_PWL[21]: x_cost_cut_PWL[21] = 0
 c_cost_PV_cut_PWL[22]: x_cost_cut_PWL[22] = 0
 c_cost_PV_cut_PWL[23]: x_cost_cut_PWL[23] = 0
 c_cost_PV_cut_PWL[24]: x_cost_cut_PWL[24] = 0
 c_cost_PV_cut_PWL[25]: x_cost_cut_PWL[25] = 0
 c_cost_PV_cut_PWL[26]: x_cost_cut_PWL[26] = 0
 c_cost_PV_cut_PWL[27]: x_cost_cut_PWL[27] = 0
 c_cost_PV_cut_PWL[28]: x_cost_cut_PWL[28] = 0
 c_cost_PV_cut_PWL[29]: x_cost_cut_PWL[29] = 0
 c_cost_PV_cut_PWL[30]: x_cost_cut_PWL[30] = 0
 c_cost_PV_cut_PWL[31]: x_cost_cut_PWL[31] = 0
 c_cost_PV_cut_PWL[32]: x_cost_cut_PWL[32] = 0
 c_cost_PV_cut_PWL[33]: x_cost_cut_PWL[33] = 0
 c_cost_PV_cut_PWL[34]: x_cost_cut_PWL[34] = 0
 c_cost_PV_cut_PWL[35]: x_cost_cut_PWL[35] = 3.919359736654292
 c_cost_PV_cut_PWL[36]: x_cost_cut_PWL[36] = 38.75579825299648
 c_cost_PV_cut_PWL[37]: x_cost_cut_PWL[37] = 109.1032859524527
 c_cost_PV_cut_PWL[38]: x_cost_cut_PWL[38] = 166.0864695716124
 c_cost_PV_cut_PWL[39]: x_cost_cut_PWL[39] = 260.3787892138844
 c_cost_PV_cut_PWL[40]: x_cost_cut_PWL[40] = 305.7469445144548
 c_cost_PV_cut_PWL[41]: x_cost_cut_PWL[41] = 345.0191343194435
 c_cost_PV_cut_PWL[42]: x_cost_cut_PWL[42] = 352.3414621609782
 c_cost_PV_cut_PWL[43]: x_cost_cut_PWL[43] = 631.0752000476556
 c_cost_PV_cut_PWL[44]: x_cost_cut_PWL[44] = 641.8824206969996
 c_cost_PV_cut_PWL[45]: x_cost_cut_PWL[45] = 682.6662040130709
 c_cost_PV_cut_PWL[46]: x_cost_cut_PWL[46] = 502.3705027905215
 c_cost_PV_cut_PWL[47]: x_cost_cut_PWL[47] = 517.7368201230703
 c_cost_PV_cut_PWL[48]: x_cost_cut_PWL[48] = 499.9846619340534
 c_cost_PV_cut_PWL[49]: x_cost_cut_PWL[49] = 561.0252146124309
 c_cost_PV_cut_PWL[50]: x_cost_cut_PWL[50] = 504.353213953263
 c_cost_PV_cut_PWL[51]: x_cost_cut_PWL[51] = 519.3484335402193
 c_cost_PV_cut_PWL[52]: x_cost_cut_PWL[52] = 497.1215462859487
 c_cost_PV_cut_PWL[53]: x_cost_cut_PWL[53] = 487.713843888524
 c_cost_PV_cut_PWL[54]: x_cost_cut_PWL[54] = 506.6190406966845
 c_cost_PV_cut_PWL[55]: x_cost_cut_PWL[55] = 527.8377702893533
 c_cost_PV_cut_PWL[56]: x_cost_cut_PWL[56] = 580.4282305412002
 c_cost_PV_cut_PWL[57]: x_cost_cut_PWL[57] = 613.1538824109894
 c_cost_PV_cut_PWL[58]: x_cost_cut_PWL[58] = 592.7357951741657
 c_cost_PV_cut_PWL[59]: x_cost_cut_PWL[59] = 683.7113245721536
 c_cost_PV_cut_PWL[60]: x_cost_cut_PWL[60] = 589.6025430859202
 c_cost_PV_cut_PWL[61]: x_cost_cut_PWL[61] = 511.0097939476314
 c_cost_PV_cut_PWL[62]: x_cost_cut_PWL[62] = 567.3924296116659
 c_cost_PV_cut_PWL[63]: x_cost_cut_PWL[63] = 436.7165367615794
 c_cost_PV_cut_PWL[64]: x_cost_cut_PWL[64] = 234.7297472521497
 c_cost_PV_cut_PWL[65]: x_cost_cut_PWL[65] = 159.9389351986541
 c_cost_PV_cut_PWL[66]: x_cost_cut_PWL[66] = 2.107330115136872
 c_cost_PV_cut_PWL[67]: x_cost_cut_PWL[67] = 2.378983563382941
 c_cost_PV_cut_PWL[68]: x_cost_cut_PWL[68] = 0
 c_cost_PV_cut_PWL[69]: x_cost_cut_PWL[69] = 0
 c_cost_PV_cut_PWL[70]: x_cost_cut_PWL[70] = 0
 c_cost_PV_cut_PWL[71]: x_cost_cut_PWL[71] = 0
 c_cost_PV_cut_PWL[72]: x_cost_cut_PWL[72] = 0
 c_cost_PV_cut_PWL[73]: x_cost_cut_PWL[73] = 0
 c_cost_PV_cut_PWL[74]: x_cost_cut_PWL[74] = 0
 c_cost_PV_cut_PWL[75]: x_cost_cut_PWL[75] = 0
 c_cost_PV_cut_PWL[76]: x_cost_cut_PWL[76] = 0
 c_cost_PV_cut_PWL[77]: x_cost_cut_PWL[77] = 0
 c_cost_PV_cut_PWL[78]: x_cost_cut_PWL[78] = 0
 c_cost_PV_cut_PWL[79]: x_cost_cut_PWL[79] = 0
 c_cost_PV_cut_PWL[80]: x_cost_cut_PWL[80] = 0
 c_cost_PV_cut_PWL[81]: x_cost_cut_PWL[81] = 0
 c_cost_PV_cut_PWL[82]: x_cost_cut_PWL[82] = 0
 c_cost_PV_cut_PWL[83]: x_cost_cut_PWL[83] = 0
 c_cost_PV_cut_PWL[84]: x_cost_cut_PWL[84] = 0
 c_cost_PV_cut_PWL[85]: x_cost_cut_PWL[85] = 0
 c_cost_PV_cut_PWL[86]: x_cost_cut_PWL[86] = 0
 c_cost_PV_cut_PWL[87]: x_cost_cut_PWL[87] = 0
 c_cost_PV_cut_PWL[88]: x_cost_cut_PWL[88] = 0
 c_cost_PV_cut_PWL[89]: x_cost_cut_PWL[89] = 0
 c_cost_PV_cut_PWL[90]: x_cost_cut_PWL[90] = 0
 c_cost_PV_cut_PWL[91]: x_cost_cut_PWL[91] = 0
 c_cost_PV_cut_PWL[92]: x_cost_cut_PWL[92] = 0
 c_cost_PV_cut_PWL[93]: x_cost_cut_PWL[93] = 0
 c_cost_PV_cut_PWL[94]: x_cost_cut_PWL[94] = 0
 c_cost_PV_cut_PWL[95]: x_cost_cut_PWL[95] = 0
 c_cost_fuel_res_re[0]: - 0.02 y_pos[0] - 0.01 y_neg[0] + y_cost_fuel[0]
   = 0
 c_cost_fuel_res_re[1]: - 0.02 y_pos[1] - 0.01 y_neg[1] + y_cost_fuel[1]
   = 0
 c_cost_fuel_res_re[2]: - 0.02 y_pos[2] - 0.01 y_neg[2] + y_cost_fuel[2]
   = 0
 c_cost_fuel_res_re[3]: - 0.02 y_pos[3] - 0.01 y_neg[3] + y_cost_fuel[3]
   = 0
 c_cost_fuel_res_re[4]: - 0.02 y_pos[4] - 0.01 y_neg[4] + y_cost_fuel[4]
   = 0
 c_cost_fuel_res_re[5]: - 0.02 y_pos[5] - 0.01 y_neg[5] + y_cost_fuel[5]
   = 0
 c_cost_fuel_res_re[6]: - 0.02 y_pos[6] - 0.01 y_neg[6] + y_cost_fuel[6]
   = 0
 c_cost_fuel_res_re[7]: - 0.02 y_pos[7] - 0.01 y_neg[7] + y_cost_fuel[7]
   = 0
 c_cost_fuel_res_re[8]: - 0.02 y_pos[8] - 0.01 y_neg[8] + y_cost_fuel[8]
   = 0
 c_cost_fuel_res_re[9]: - 0.02 y_pos[9] - 0.01 y_neg[9] + y_cost_fuel[9]
   = 0
 c_cost_fuel_res_re[10]: - 0.02 y_pos[10] - 0.01 y_neg[10]
   + y_cost_fuel[10] = 0
 c_cost_fuel_res_re[11]: - 0.02 y_pos[11] - 0.01 y_neg[11]
   + y_cost_fuel[11] = 0
 c_cost_fuel_res_re[12]: - 0.02 y_pos[12] - 0.01 y_neg[12]
   + y_cost_fuel[12] = 0
 c_cost_fuel_res_re[13]: - 0.02 y_pos[13] - 0.01 y_neg[13]
   + y_cost_fuel[13] = 0
 c_cost_fuel_res_re[14]: - 0.02 y_pos[14] - 0.01 y_neg[14]
   + y_cost_fuel[14] = 0
 c_cost_fuel_res_re[15]: - 0.02 y_pos[15] - 0.01 y_neg[15]
   + y_cost_fuel[15] = 0
 c_cost_fuel_res_re[16]: - 0.02 y_pos[16] - 0.01 y_neg[16]
   + y_cost_fuel[16] = 0
 c_cost_fuel_res_re[17]: - 0.02 y_pos[17] - 0.01 y_neg[17]
   + y_cost_fuel[17] = 0
 c_cost_fuel_res_re[18]: - 0.02 y_pos[18] - 0.01 y_neg[18]
   + y_cost_fuel[18] = 0
 c_cost_fuel_res_re[19]: - 0.02 y_pos[19] - 0.01 y_neg[19]
   + y_cost_fuel[19] = 0
 c_cost_fuel_res_re[20]: - 0.02 y_pos[20] - 0.01 y_neg[20]
   + y_cost_fuel[20] = 0
 c_cost_fuel_res_re[21]: - 0.02 y_pos[21] - 0.01 y_neg[21]
   + y_cost_fuel[21] = 0
 c_cost_fuel_res_re[22]: - 0.02 y_pos[22] - 0.01 y_neg[22]
   + y_cost_fuel[22] = 0
 c_cost_fuel_res_re[23]: - 0.02 y_pos[23] - 0.01 y_neg[23]
   + y_cost_fuel[23] = 0
 c_cost_fuel_res_re[24]: - 0.02 y_pos[24] - 0.01 y_neg[24]
   + y_cost_fuel[24] = 0
 c_cost_fuel_res_re[25]: - 0.02 y_pos[25] - 0.01 y_neg[25]
   + y_cost_fuel[25] = 0
 c_cost_fuel_res_re[26]: - 0.02 y_pos[26] - 0.01 y_neg[26]
   + y_cost_fuel[26] = 0
 c_cost_fuel_res_re[27]: - 0.02 y_pos[27] - 0.01 y_neg[27]
   + y_cost_fuel[27] = 0
 c_cost_fuel_res_re[28]: - 0.02 y_pos[28] - 0.01 y_neg[28]
   + y_cost_fuel[28] = 0
 c_cost_fuel_res_re[29]: - 0.02 y_pos[29] - 0.01 y_neg[29]
   + y_cost_fuel[29] = 0
 c_cost_fuel_res_re[30]: - 0.02 y_pos[30] - 0.01 y_neg[30]
   + y_cost_fuel[30] = 0
 c_cost_fuel_res_re[31]: - 0.02 y_pos[31] - 0.01 y_neg[31]
   + y_cost_fuel[31] = 0
 c_cost_fuel_res_re[32]: - 0.02 y_pos[32] - 0.01 y_neg[32]
   + y_cost_fuel[32] = 0
 c_cost_fuel_res_re[33]: - 0.02 y_pos[33] - 0.01 y_neg[33]
   + y_cost_fuel[33] = 0
 c_cost_fuel_res_re[34]: - 0.02 y_pos[34] - 0.01 y_neg[34]
   + y_cost_fuel[34] = 0
 c_cost_fuel_res_re[35]: - 0.02 y_pos[35] - 0.01 y_neg[35]
   + y_cost_fuel[35] = 0
 c_cost_fuel_res_re[36]: - 0.02 y_pos[36] - 0.01 y_neg[36]
   + y_cost_fuel[36] = 0
 c_cost_fuel_res_re[37]: - 0.02 y_pos[37] - 0.01 y_neg[37]
   + y_cost_fuel[37] = 0
 c_cost_fuel_res_re[38]: - 0.02 y_pos[38] - 0.01 y_neg[38]
   + y_cost_fuel[38] = 0
 c_cost_fuel_res_re[39]: - 0.02 y_pos[39] - 0.01 y_neg[39]
   + y_cost_fuel[39] = 0
 c_cost_fuel_res_re[40]: - 0.02 y_pos[40] - 0.01 y_neg[40]
   + y_cost_fuel[40] = 0
 c_cost_fuel_res_re[41]: - 0.02 y_pos[41] - 0.01 y_neg[41]
   + y_cost_fuel[41] = 0
 c_cost_fuel_res_re[42]: - 0.02 y_pos[42] - 0.01 y_neg[42]
   + y_cost_fuel[42] = 0
 c_cost_fuel_res_re[43]: - 0.02 y_pos[43] - 0.01 y_neg[43]
   + y_cost_fuel[43] = 0
 c_cost_fuel_res_re[44]: - 0.02 y_pos[44] - 0.01 y_neg[44]
   + y_cost_fuel[44] = 0
 c_cost_fuel_res_re[45]: - 0.02 y_pos[45] - 0.01 y_neg[45]
   + y_cost_fuel[45] = 0
 c_cost_fuel_res_re[46]: - 0.02 y_pos[46] - 0.01 y_neg[46]
   + y_cost_fuel[46] = 0
 c_cost_fuel_res_re[47]: - 0.02 y_pos[47] - 0.01 y_neg[47]
   + y_cost_fuel[47] = 0
 c_cost_fuel_res_re[48]: - 0.02 y_pos[48] - 0.01 y_neg[48]
   + y_cost_fuel[48] = 0
 c_cost_fuel_res_re[49]: - 0.02 y_pos[49] - 0.01 y_neg[49]
   + y_cost_fuel[49] = 0
 c_cost_fuel_res_re[50]: - 0.02 y_pos[50] - 0.01 y_neg[50]
   + y_cost_fuel[50] = 0
 c_cost_fuel_res_re[51]: - 0.02 y_pos[51] - 0.01 y_neg[51]
   + y_cost_fuel[51] = 0
 c_cost_fuel_res_re[52]: - 0.02 y_pos[52] - 0.01 y_neg[52]
   + y_cost_fuel[52] = 0
 c_cost_fuel_res_re[53]: - 0.02 y_pos[53] - 0.01 y_neg[53]
   + y_cost_fuel[53] = 0
 c_cost_fuel_res_re[54]: - 0.02 y_pos[54] - 0.01 y_neg[54]
   + y_cost_fuel[54] = 0
 c_cost_fuel_res_re[55]: - 0.02 y_pos[55] - 0.01 y_neg[55]
   + y_cost_fuel[55] = 0
 c_cost_fuel_res_re[56]: - 0.02 y_pos[56] - 0.01 y_neg[56]
   + y_cost_fuel[56] = 0
 c_cost_fuel_res_re[57]: - 0.02 y_pos[57] - 0.01 y_neg[57]
   + y_cost_fuel[57] = 0
 c_cost_fuel_res_re[58]: - 0.02 y_pos[58] - 0.01 y_neg[58]
   + y_cost_fuel[58] = 0
 c_cost_fuel_res_re[59]: - 0.02 y_pos[59] - 0.01 y_neg[59]
   + y_cost_fuel[59] = 0
 c_cost_fuel_res_re[60]: - 0.02 y_pos[60] - 0.01 y_neg[60]
   + y_cost_fuel[60] = 0
 c_cost_fuel_res_re[61]: - 0.02 y_pos[61] - 0.01 y_neg[61]
   + y_cost_fuel[61] = 0
 c_cost_fuel_res_re[62]: - 0.02 y_pos[62] - 0.01 y_neg[62]
   + y_cost_fuel[62] = 0
 c_cost_fuel_res_re[63]: - 0.02 y_pos[63] - 0.01 y_neg[63]
   + y_cost_fuel[63] = 0
 c_cost_fuel_res_re[64]: - 0.02 y_pos[64] - 0.01 y_neg[64]
   + y_cost_fuel[64] = 0
 c_cost_fuel_res_re[65]: - 0.02 y_pos[65] - 0.01 y_neg[65]
   + y_cost_fuel[65] = 0
 c_cost_fuel_res_re[66]: - 0.02 y_pos[66] - 0.01 y_neg[66]
   + y_cost_fuel[66] = 0
 c_cost_fuel_res_re[67]: - 0.02 y_pos[67] - 0.01 y_neg[67]
   + y_cost_fuel[67] = 0
 c_cost_fuel_res_re[68]: - 0.02 y_pos[68] - 0.01 y_neg[68]
   + y_cost_fuel[68] = 0
 c_cost_fuel_res_re[69]: - 0.02 y_pos[69] - 0.01 y_neg[69]
   + y_cost_fuel[69] = 0
 c_cost_fuel_res_re[70]: - 0.02 y_pos[70] - 0.01 y_neg[70]
   + y_cost_fuel[70] = 0
 c_cost_fuel_res_re[71]: - 0.02 y_pos[71] - 0.01 y_neg[71]
   + y_cost_fuel[71] = 0
 c_cost_fuel_res_re[72]: - 0.02 y_pos[72] - 0.01 y_neg[72]
   + y_cost_fuel[72] = 0
 c_cost_fuel_res_re[73]: - 0.02 y_pos[73] - 0.01 y_neg[73]
   + y_cost_fuel[73] = 0
 c_cost_fuel_res_re[74]: - 0.02 y_pos[74] - 0.01 y_neg[74]
   + y_cost_fuel[74] = 0
 c_cost_fuel_res_re[75]: - 0.02 y_pos[75] - 0.01 y_neg[75]
   + y_cost_fuel[75] = 0
 c_cost_fuel_res_re[76]: - 0.02 y_pos[76] - 0.01 y_neg[76]
   + y_cost_fuel[76] = 0
 c_cost_fuel_res_re[77]: - 0.02 y_pos[77] - 0.01 y_neg[77]
   + y_cost_fuel[77] = 0
 c_cost_fuel_res_re[78]: - 0.02 y_pos[78] - 0.01 y_neg[78]
   + y_cost_fuel[78] = 0
 c_cost_fuel_res_re[79]: - 0.02 y_pos[79] - 0.01 y_neg[79]
   + y_cost_fuel[79] = 0
 c_cost_fuel_res_re[80]: - 0.02 y_pos[80] - 0.01 y_neg[80]
   + y_cost_fuel[80] = 0
 c_cost_fuel_res_re[81]: - 0.02 y_pos[81] - 0.01 y_neg[81]
   + y_cost_fuel[81] = 0
 c_cost_fuel_res_re[82]: - 0.02 y_pos[82] - 0.01 y_neg[82]
   + y_cost_fuel[82] = 0
 c_cost_fuel_res_re[83]: - 0.02 y_pos[83] - 0.01 y_neg[83]
   + y_cost_fuel[83] = 0
 c_cost_fuel_res_re[84]: - 0.02 y_pos[84] - 0.01 y_neg[84]
   + y_cost_fuel[84] = 0
 c_cost_fuel_res_re[85]: - 0.02 y_pos[85] - 0.01 y_neg[85]
   + y_cost_fuel[85] = 0
 c_cost_fuel_res_re[86]: - 0.02 y_pos[86] - 0.01 y_neg[86]
   + y_cost_fuel[86] = 0
 c_cost_fuel_res_re[87]: - 0.02 y_pos[87] - 0.01 y_neg[87]
   + y_cost_fuel[87] = 0
 c_cost_fuel_res_re[88]: - 0.02 y_pos[88] - 0.01 y_neg[88]
   + y_cost_fuel[88] = 0
 c_cost_fuel_res_re[89]: - 0.02 y_pos[89] - 0.01 y_neg[89]
   + y_cost_fuel[89] = 0
 c_cost_fuel_res_re[90]: - 0.02 y_pos[90] - 0.01 y_neg[90]
   + y_cost_fuel[90] = 0
 c_cost_fuel_res_re[91]: - 0.02 y_pos[91] - 0.01 y_neg[91]
   + y_cost_fuel[91] = 0
 c_cost_fuel_res_re[92]: - 0.02 y_pos[92] - 0.01 y_neg[92]
   + y_cost_fuel[92] = 0
 c_cost_fuel_res_re[93]: - 0.02 y_pos[93] - 0.01 y_neg[93]
   + y_cost_fuel[93] = 0
 c_cost_fuel_res_re[94]: - 0.02 y_pos[94] - 0.01 y_neg[94]
   + y_cost_fuel[94] = 0
 c_cost_fuel_res_re[95]: - 0.02 y_pos[95] - 0.01 y_neg[95]
   + y_cost_fuel[95] = 0
 c_cost_ESS_OM_re[0]: - 0.005 y_chg[0] - 0.005 y_dis[0] + y_cost_ESS[0]
   = 0
 c_cost_ESS_OM_re[1]: - 0.005 y_chg[1] - 0.005 y_dis[1] + y_cost_ESS[1]
   = 0
 c_cost_ESS_OM_re[2]: - 0.005 y_chg[2] - 0.005 y_dis[2] + y_cost_ESS[2]
   = 0
 c_cost_ESS_OM_re[3]: - 0.005 y_chg[3] - 0.005 y_dis[3] + y_cost_ESS[3]
   = 0
 c_cost_ESS_OM_re[4]: - 0.005 y_chg[4] - 0.005 y_dis[4] + y_cost_ESS[4]
   = 0
 c_cost_ESS_OM_re[5]: - 0.005 y_chg[5] - 0.005 y_dis[5] + y_cost_ESS[5]
   = 0
 c_cost_ESS_OM_re[6]: - 0.005 y_chg[6] - 0.005 y_dis[6] + y_cost_ESS[6]
   = 0
 c_cost_ESS_OM_re[7]: - 0.005 y_chg[7] - 0.005 y_dis[7] + y_cost_ESS[7]
   = 0
 c_cost_ESS_OM_re[8]: - 0.005 y_chg[8] - 0.005 y_dis[8] + y_cost_ESS[8]
   = 0
 c_cost_ESS_OM_re[9]: - 0.005 y_chg[9] - 0.005 y_dis[9] + y_cost_ESS[9]
   = 0
 c_cost_ESS_OM_re[10]: - 0.005 y_chg[10] - 0.005 y_dis[10] + y_cost_ESS[10]
   = 0
 c_cost_ESS_OM_re[11]: - 0.005 y_chg[11] - 0.005 y_dis[11] + y_cost_ESS[11]
   = 0
 c_cost_ESS_OM_re[12]: - 0.005 y_chg[12] - 0.005 y_dis[12] + y_cost_ESS[12]
   = 0
 c_cost_ESS_OM_re[13]: - 0.005 y_chg[13] - 0.005 y_dis[13] + y_cost_ESS[13]
   = 0
 c_cost_ESS_OM_re[14]: - 0.005 y_chg[14] - 0.005 y_dis[14] + y_cost_ESS[14]
   = 0
 c_cost_ESS_OM_re[15]: - 0.005 y_chg[15] - 0.005 y_dis[15] + y_cost_ESS[15]
   = 0
 c_cost_ESS_OM_re[16]: - 0.005 y_chg[16] - 0.005 y_dis[16] + y_cost_ESS[16]
   = 0
 c_cost_ESS_OM_re[17]: - 0.005 y_chg[17] - 0.005 y_dis[17] + y_cost_ESS[17]
   = 0
 c_cost_ESS_OM_re[18]: - 0.005 y_chg[18] - 0.005 y_dis[18] + y_cost_ESS[18]
   = 0
 c_cost_ESS_OM_re[19]: - 0.005 y_chg[19] - 0.005 y_dis[19] + y_cost_ESS[19]
   = 0
 c_cost_ESS_OM_re[20]: - 0.005 y_chg[20] - 0.005 y_dis[20] + y_cost_ESS[20]
   = 0
 c_cost_ESS_OM_re[21]: - 0.005 y_chg[21] - 0.005 y_dis[21] + y_cost_ESS[21]
   = 0
 c_cost_ESS_OM_re[22]: - 0.005 y_chg[22] - 0.005 y_dis[22] + y_cost_ESS[22]
   = 0
 c_cost_ESS_OM_re[23]: - 0.005 y_chg[23] - 0.005 y_dis[23] + y_cost_ESS[23]
   = 0
 c_cost_ESS_OM_re[24]: - 0.005 y_chg[24] - 0.005 y_dis[24] + y_cost_ESS[24]
   = 0
 c_cost_ESS_OM_re[25]: - 0.005 y_chg[25] - 0.005 y_dis[25] + y_cost_ESS[25]
   = 0
 c_cost_ESS_OM_re[26]: - 0.005 y_chg[26] - 0.005 y_dis[26] + y_cost_ESS[26]
   = 0
 c_cost_ESS_OM_re[27]: - 0.005 y_chg[27] - 0.005 y_dis[27] + y_cost_ESS[27]
   = 0
 c_cost_ESS_OM_re[28]: - 0.005 y_chg[28] - 0.005 y_dis[28] + y_cost_ESS[28]
   = 0
 c_cost_ESS_OM_re[29]: - 0.005 y_chg[29] - 0.005 y_dis[29] + y_cost_ESS[29]
   = 0
 c_cost_ESS_OM_re[30]: - 0.005 y_chg[30] - 0.005 y_dis[30] + y_cost_ESS[30]
   = 0
 c_cost_ESS_OM_re[31]: - 0.005 y_chg[31] - 0.005 y_dis[31] + y_cost_ESS[31]
   = 0
 c_cost_ESS_OM_re[32]: - 0.005 y_chg[32] - 0.005 y_dis[32] + y_cost_ESS[32]
   = 0
 c_cost_ESS_OM_re[33]: - 0.005 y_chg[33] - 0.005 y_dis[33] + y_cost_ESS[33]
   = 0
 c_cost_ESS_OM_re[34]: - 0.005 y_chg[34] - 0.005 y_dis[34] + y_cost_ESS[34]
   = 0
 c_cost_ESS_OM_re[35]: - 0.005 y_chg[35] - 0.005 y_dis[35] + y_cost_ESS[35]
   = 0
 c_cost_ESS_OM_re[36]: - 0.005 y_chg[36] - 0.005 y_dis[36] + y_cost_ESS[36]
   = 0
 c_cost_ESS_OM_re[37]: - 0.005 y_chg[37] - 0.005 y_dis[37] + y_cost_ESS[37]
   = 0
 c_cost_ESS_OM_re[38]: - 0.005 y_chg[38] - 0.005 y_dis[38] + y_cost_ESS[38]
   = 0
 c_cost_ESS_OM_re[39]: - 0.005 y_chg[39] - 0.005 y_dis[39] + y_cost_ESS[39]
   = 0
 c_cost_ESS_OM_re[40]: - 0.005 y_chg[40] - 0.005 y_dis[40] + y_cost_ESS[40]
   = 0
 c_cost_ESS_OM_re[41]: - 0.005 y_chg[41] - 0.005 y_dis[41] + y_cost_ESS[41]
   = 0
 c_cost_ESS_OM_re[42]: - 0.005 y_chg[42] - 0.005 y_dis[42] + y_cost_ESS[42]
   = 0
 c_cost_ESS_OM_re[43]: - 0.005 y_chg[43] - 0.005 y_dis[43] + y_cost_ESS[43]
   = 0
 c_cost_ESS_OM_re[44]: - 0.005 y_chg[44] - 0.005 y_dis[44] + y_cost_ESS[44]
   = 0
 c_cost_ESS_OM_re[45]: - 0.005 y_chg[45] - 0.005 y_dis[45] + y_cost_ESS[45]
   = 0
 c_cost_ESS_OM_re[46]: - 0.005 y_chg[46] - 0.005 y_dis[46] + y_cost_ESS[46]
   = 0
 c_cost_ESS_OM_re[47]: - 0.005 y_chg[47] - 0.005 y_dis[47] + y_cost_ESS[47]
   = 0
 c_cost_ESS_OM_re[48]: - 0.005 y_chg[48] - 0.005 y_dis[48] + y_cost_ESS[48]
   = 0
 c_cost_ESS_OM_re[49]: - 0.005 y_chg[49] - 0.005 y_dis[49] + y_cost_ESS[49]
   = 0
 c_cost_ESS_OM_re[50]: - 0.005 y_chg[50] - 0.005 y_dis[50] + y_cost_ESS[50]
   = 0
 c_cost_ESS_OM_re[51]: - 0.005 y_chg[51] - 0.005 y_dis[51] + y_cost_ESS[51]
   = 0
 c_cost_ESS_OM_re[52]: - 0.005 y_chg[52] - 0.005 y_dis[52] + y_cost_ESS[52]
   = 0
 c_cost_ESS_OM_re[53]: - 0.005 y_chg[53] - 0.005 y_dis[53] + y_cost_ESS[53]
   = 0
 c_cost_ESS_OM_re[54]: - 0.005 y_chg[54] - 0.005 y_dis[54] + y_cost_ESS[54]
   = 0
 c_cost_ESS_OM_re[55]: - 0.005 y_chg[55] - 0.005 y_dis[55] + y_cost_ESS[55]
   = 0
 c_cost_ESS_OM_re[56]: - 0.005 y_chg[56] - 0.005 y_dis[56] + y_cost_ESS[56]
   = 0
 c_cost_ESS_OM_re[57]: - 0.005 y_chg[57] - 0.005 y_dis[57] + y_cost_ESS[57]
   = 0
 c_cost_ESS_OM_re[58]: - 0.005 y_chg[58] - 0.005 y_dis[58] + y_cost_ESS[58]
   = 0
 c_cost_ESS_OM_re[59]: - 0.005 y_chg[59] - 0.005 y_dis[59] + y_cost_ESS[59]
   = 0
 c_cost_ESS_OM_re[60]: - 0.005 y_chg[60] - 0.005 y_dis[60] + y_cost_ESS[60]
   = 0
 c_cost_ESS_OM_re[61]: - 0.005 y_chg[61] - 0.005 y_dis[61] + y_cost_ESS[61]
   = 0
 c_cost_ESS_OM_re[62]: - 0.005 y_chg[62] - 0.005 y_dis[62] + y_cost_ESS[62]
   = 0
 c_cost_ESS_OM_re[63]: - 0.005 y_chg[63] - 0.005 y_dis[63] + y_cost_ESS[63]
   = 0
 c_cost_ESS_OM_re[64]: - 0.005 y_chg[64] - 0.005 y_dis[64] + y_cost_ESS[64]
   = 0
 c_cost_ESS_OM_re[65]: - 0.005 y_chg[65] - 0.005 y_dis[65] + y_cost_ESS[65]
   = 0
 c_cost_ESS_OM_re[66]: - 0.005 y_chg[66] - 0.005 y_dis[66] + y_cost_ESS[66]
   = 0
 c_cost_ESS_OM_re[67]: - 0.005 y_chg[67] - 0.005 y_dis[67] + y_cost_ESS[67]
   = 0
 c_cost_ESS_OM_re[68]: - 0.005 y_chg[68] - 0.005 y_dis[68] + y_cost_ESS[68]
   = 0
 c_cost_ESS_OM_re[69]: - 0.005 y_chg[69] - 0.005 y_dis[69] + y_cost_ESS[69]
   = 0
 c_cost_ESS_OM_re[70]: - 0.005 y_chg[70] - 0.005 y_dis[70] + y_cost_ESS[70]
   = 0
 c_cost_ESS_OM_re[71]: - 0.005 y_chg[71] - 0.005 y_dis[71] + y_cost_ESS[71]
   = 0
 c_cost_ESS_OM_re[72]: - 0.005 y_chg[72] - 0.005 y_dis[72] + y_cost_ESS[72]
   = 0
 c_cost_ESS_OM_re[73]: - 0.005 y_chg[73] - 0.005 y_dis[73] + y_cost_ESS[73]
   = 0
 c_cost_ESS_OM_re[74]: - 0.005 y_chg[74] - 0.005 y_dis[74] + y_cost_ESS[74]
   = 0
 c_cost_ESS_OM_re[75]: - 0.005 y_chg[75] - 0.005 y_dis[75] + y_cost_ESS[75]
   = 0
 c_cost_ESS_OM_re[76]: - 0.005 y_chg[76] - 0.005 y_dis[76] + y_cost_ESS[76]
   = 0
 c_cost_ESS_OM_re[77]: - 0.005 y_chg[77] - 0.005 y_dis[77] + y_cost_ESS[77]
   = 0
 c_cost_ESS_OM_re[78]: - 0.005 y_chg[78] - 0.005 y_dis[78] + y_cost_ESS[78]
   = 0
 c_cost_ESS_OM_re[79]: - 0.005 y_chg[79] - 0.005 y_dis[79] + y_cost_ESS[79]
   = 0
 c_cost_ESS_OM_re[80]: - 0.005 y_chg[80] - 0.005 y_dis[80] + y_cost_ESS[80]
   = 0
 c_cost_ESS_OM_re[81]: - 0.005 y_chg[81] - 0.005 y_dis[81] + y_cost_ESS[81]
   = 0
 c_cost_ESS_OM_re[82]: - 0.005 y_chg[82] - 0.005 y_dis[82] + y_cost_ESS[82]
   = 0
 c_cost_ESS_OM_re[83]: - 0.005 y_chg[83] - 0.005 y_dis[83] + y_cost_ESS[83]
   = 0
 c_cost_ESS_OM_re[84]: - 0.005 y_chg[84] - 0.005 y_dis[84] + y_cost_ESS[84]
   = 0
 c_cost_ESS_OM_re[85]: - 0.005 y_chg[85] - 0.005 y_dis[85] + y_cost_ESS[85]
   = 0
 c_cost_ESS_OM_re[86]: - 0.005 y_chg[86] - 0.005 y_dis[86] + y_cost_ESS[86]
   = 0
 c_cost_ESS_OM_re[87]: - 0.005 y_chg[87] - 0.005 y_dis[87] + y_cost_ESS[87]
   = 0
 c_cost_ESS_OM_re[88]: - 0.005 y_chg[88] - 0.005 y_dis[88] + y_cost_ESS[88]
   = 0
 c_cost_ESS_OM_re[89]: - 0.005 y_chg[89] - 0.005 y_dis[89] + y_cost_ESS[89]
   = 0
 c_cost_ESS_OM_re[90]: - 0.005 y_chg[90] - 0.005 y_dis[90] + y_cost_ESS[90]
   = 0
 c_cost_ESS_OM_re[91]: - 0.005 y_chg[91] - 0.005 y_dis[91] + y_cost_ESS[91]
   = 0
 c_cost_ESS_OM_re[92]: - 0.005 y_chg[92] - 0.005 y_dis[92] + y_cost_ESS[92]
   = 0
 c_cost_ESS_OM_re[93]: - 0.005 y_chg[93] - 0.005 y_dis[93] + y_cost_ESS[93]
   = 0
 c_cost_ESS_OM_re[94]: - 0.005 y_chg[94] - 0.005 y_dis[94] + y_cost_ESS[94]
   = 0
 c_cost_ESS_OM_re[95]: - 0.005 y_chg[95] - 0.005 y_dis[95] + y_cost_ESS[95]
   = 0
 c_reserve_pos[0]: y_pos[0] <= 0
 c_reserve_pos[1]: y_pos[1] <= 0
 c_reserve_pos[2]: y_pos[2] <= 0
 c_reserve_pos[3]: y_pos[3] <= 0
 c_reserve_pos[4]: y_pos[4] <= 0
 c_reserve_pos[5]: y_pos[5] <= 0
 c_reserve_pos[6]: y_pos[6] <= 0
 c_reserve_pos[7]: y_pos[7] <= 0
 c_reserve_pos[8]: y_pos[8] <= 0
 c_reserve_pos[9]: y_pos[9] <= 0
 c_reserve_pos[10]: y_pos[10] <= 0
 c_reserve_pos[11]: y_pos[11] <= 0
 c_reserve_pos[12]: y_pos[12] <= 0
 c_reserve_pos[13]: y_pos[13] <= 0
 c_reserve_pos[14]: y_pos[14] <= 0
 c_reserve_pos[15]: y_pos[15] <= 0
 c_reserve_pos[16]: y_pos[16] <= 0
 c_reserve_pos[17]: y_pos[17] <= 0
 c_reserve_pos[18]: y_pos[18] <= 0
 c_reserve_pos[19]: y_pos[19] <= 0
 c_reserve_pos[20]: y_pos[20] <= 0
 c_reserve_pos[21]: y_pos[21] <= 0
 c_reserve_pos[22]: y_pos[22] <= 0
 c_reserve_pos[23]: y_pos[23] <= 0
 c_reserve_pos[24]: y_pos[24] <= 0
 c_reserve_pos[25]: y_pos[25] <= 0
 c_reserve_pos[26]: y_pos[26] <= 0
 c_reserve_pos[27]: y_pos[27] <= 0
 c_reserve_pos[28]: y_pos[28] <= 0
 c_reserve_pos[29]: y_pos[29] <= 0
 c_reserve_pos[30]: y_pos[30] <= 0
 c_reserve_pos[31]: y_pos[31] <= 0
 c_reserve_pos[32]: y_pos[32] <= 0
 c_reserve_pos[33]: y_pos[33] <= 0
 c_reserve_pos[34]: y_pos[34] <= 0
 c_reserve_pos[35]: y_pos[35] <= 0
 c_reserve_pos[36]: y_pos[36] <= 0
 c_reserve_pos[37]: y_pos[37] <= 0
 c_reserve_pos[38]: y_pos[38] <= 0
 c_reserve_pos[39]: y_pos[39] <= 0
 c_reserve_pos[40]: y_pos[40] <= 0
 c_reserve_pos[41]: y_pos[41] <= 0
 c_reserve_pos[42]: y_pos[42] <= 0
 c_reserve_pos[43]: y_pos[43] <= 0
 c_reserve_pos[44]: y_pos[44] <= 0
 c_reserve_pos[45]: y_pos[45] <= 0
 c_reserve_pos[46]: y_pos[46] <= 0
 c_reserve_pos[47]: y_pos[47] <= 0
 c_reserve_pos[48]: y_pos[48] <= 0
 c_reserve_pos[49]: y_pos[49] <= 0
 c_reserve_pos[50]: y_pos[50] <= 0
 c_reserve_pos[51]: y_pos[51] <= 0
 c_reserve_pos[52]: y_pos[52] <= 0
 c_reserve_pos[53]: y_pos[53] <= 0
 c_reserve_pos[54]: y_pos[54] <= 0
 c_reserve_pos[55]: y_pos[55] <= 0
 c_reserve_pos[56]: y_pos[56] <= 0
 c_reserve_pos[57]: y_pos[57] <= 0
 c_reserve_pos[58]: y_pos[58] <= 0
 c_reserve_pos[59]: y_pos[59] <= 0
 c_reserve_pos[60]: y_pos[60] <= 0
 c_reserve_pos[61]: y_pos[61] <= 0
 c_reserve_pos[62]: y_pos[62] <= 0
 c_reserve_pos[63]: y_pos[63] <= 0
 c_reserve_pos[64]: y_pos[64] <= 0
 c_reserve_pos[65]: y_pos[65] <= 0
 c_reserve_pos[66]: y_pos[66] <= 0
 c_reserve_pos[67]: y_pos[67] <= 0
 c_reserve_pos[68]: y_pos[68] <= 0
 c_reserve_pos[69]: y_pos[69] <= 0
 c_reserve_pos[70]: y_pos[70] <= 0
 c_reserve_pos[71]: y_pos[71] <= 0
 c_reserve_pos[72]: y_pos[72] <= 0
 c_reserve_pos[73]: y_pos[73] <= 0
 c_reserve_pos[74]: y_pos[74] <= 0
 c_reserve_pos[75]: y_pos[75] <= 0
 c_reserve_pos[76]: y_pos[76] <= 0
 c_reserve_pos[77]: y_pos[77] <= 0
 c_reserve_pos[78]: y_pos[78] <= 0
 c_reserve_pos[79]: y_pos[79] <= 0
 c_reserve_pos[80]: y_pos[80] <= 0
 c_reserve_pos[81]: y_pos[81] <= 0
 c_reserve_pos[82]: y_pos[82] <= 0
 c_reserve_pos[83]: y_pos[83] <= 0
 c_reserve_pos[84]: y_pos[84] <= 0
 c_reserve_pos[85]: y_pos[85] <= 0
 c_reserve_pos[86]: y_pos[86] <= 0
 c_reserve_pos[87]: y_pos[87] <= 0
 c_reserve_pos[88]: y_pos[88] <= 0
 c_reserve_pos[89]: y_pos[89] <= 0
 c_reserve_pos[90]: y_pos[90] <= 0
 c_reserve_pos[91]: y_pos[91] <= 0
 c_reserve_pos[92]: y_pos[92] <= 0
 c_reserve_pos[93]: y_pos[93] <= 0
 c_reserve_pos[94]: y_pos[94] <= 0
 c_reserve_pos[95]: y_pos[95] <= 0
 c_reserve_neg[0]: y_neg[0] <= 0
 c_reserve_neg[1]: y_neg[1] <= 0
 c_reserve_neg[2]: y_neg[2] <= 0
 c_reserve_neg[3]: y_neg[3] <= 0
 c_reserve_neg[4]: y_neg[4] <= 0
 c_reserve_neg[5]: y_neg[5] <= 0
 c_reserve_neg[6]: y_neg[6] <= 0
 c_reserve_neg[7]: y_neg[7] <= 0
 c_reserve_neg[8]: y_neg[8] <= 0
 c_reserve_neg[9]: y_neg[9] <= 0
 c_reserve_neg[10]: y_neg[10] <= 0
 c_reserve_neg[11]: y_neg[11] <= 0
 c_reserve_neg[12]: y_neg[12] <= 0
 c_reserve_neg[13]: y_neg[13] <= 0
 c_reserve_neg[14]: y_neg[14] <= 0
 c_reserve_neg[15]: y_neg[15] <= 0
 c_reserve_neg[16]: y_neg[16] <= 0
 c_reserve_neg[17]: y_neg[17] <= 0
 c_reserve_neg[18]: y_neg[18] <= 0
 c_reserve_neg[19]: y_neg[19] <= 0
 c_reserve_neg[20]: y_neg[20] <= 0
 c_reserve_neg[21]: y_neg[21] <= 0
 c_reserve_neg[22]: y_neg[22] <= 0
 c_reserve_neg[23]: y_neg[23] <= 0
 c_reserve_neg[24]: y_neg[24] <= 0
 c_reserve_neg[25]: y_neg[25] <= 0
 c_reserve_neg[26]: y_neg[26] <= 0
 c_reserve_neg[27]: y_neg[27] <= 0
 c_reserve_neg[28]: y_neg[28] <= 0
 c_reserve_neg[29]: y_neg[29] <= 0
 c_reserve_neg[30]: y_neg[30] <= 0
 c_reserve_neg[31]: y_neg[31] <= 0
 c_reserve_neg[32]: y_neg[32] <= 0
 c_reserve_neg[33]: y_neg[33] <= 0
 c_reserve_neg[34]: y_neg[34] <= 0
 c_reserve_neg[35]: y_neg[35] <= 0
 c_reserve_neg[36]: y_neg[36] <= 0
 c_reserve_neg[37]: y_neg[37] <= 0
 c_reserve_neg[38]: y_neg[38] <= 0
 c_reserve_neg[39]: y_neg[39] <= 0
 c_reserve_neg[40]: y_neg[40] <= 0
 c_reserve_neg[41]: y_neg[41] <= 0
 c_reserve_neg[42]: y_neg[42] <= 0
 c_reserve_neg[43]: y_neg[43] <= 0
 c_reserve_neg[44]: y_neg[44] <= 0
 c_reserve_neg[45]: y_neg[45] <= 0
 c_reserve_neg[46]: y_neg[46] <= 0
 c_reserve_neg[47]: y_neg[47] <= 0
 c_reserve_neg[48]: y_neg[48] <= 0
 c_reserve_neg[49]: y_neg[49] <= 0
 c_reserve_neg[50]: y_neg[50] <= 0
 c_reserve_neg[51]: y_neg[51] <= 0
 c_reserve_neg[52]: y_neg[52] <= 0
 c_reserve_neg[53]: y_neg[53] <= 0
 c_reserve_neg[54]: y_neg[54] <= 0
 c_reserve_neg[55]: y_neg[55] <= 0
 c_reserve_neg[56]: y_neg[56] <= 0
 c_reserve_neg[57]: y_neg[57] <= 0
 c_reserve_neg[58]: y_neg[58] <= 0
 c_reserve_neg[59]: y_neg[59] <= 0
 c_reserve_neg[60]: y_neg[60] <= 0
 c_reserve_neg[61]: y_neg[61] <= 0
 c_reserve_neg[62]: y_neg[62] <= 0
 c_reserve_neg[63]: y_neg[63] <= 0
 c_reserve_neg[64]: y_neg[64] <= 0
 c_reserve_neg[65]: y_neg[65] <= 0
 c_reserve_neg[66]: y_neg[66] <= 0
 c_reserve_neg[67]: y_neg[67] <= 0
 c_reserve_neg[68]: y_neg[68] <= 0
 c_reserve_neg[69]: y_neg[69] <= 0
 c_reserve_neg[70]: y_neg[70] <= 0
 c_reserve_neg[71]: y_neg[71] <= 0
 c_reserve_neg[72]: y_neg[72] <= 0
 c_reserve_neg[73]: y_neg[73] <= 0
 c_reserve_neg[74]: y_neg[74] <= 0
 c_reserve_neg[75]: y_neg[75] <= 0
 c_reserve_neg[76]: y_neg[76] <= 0
 c_reserve_neg[77]: y_neg[77] <= 0
 c_reserve_neg[78]: y_neg[78] <= 0
 c_reserve_neg[79]: y_neg[79] <= 0
 c_reserve_neg[80]: y_neg[80] <= 0
 c_reserve_neg[81]: y_neg[81] <= 0
 c_reserve_neg[82]: y_neg[82] <= 0
 c_reserve_neg[83]: y_neg[83] <= 0
 c_reserve_neg[84]: y_neg[84] <= 0
 c_reserve_neg[85]: y_neg[85] <= 0
 c_reserve_neg[86]: y_neg[86] <= 0
 c_reserve_neg[87]: y_neg[87] <= 0
 c_reserve_neg[88]: y_neg[88] <= 0
 c_reserve_neg[89]: y_neg[89] <= 0
 c_reserve_neg[90]: y_neg[90] <= 0
 c_reserve_neg[91]: y_neg[91] <= 0
 c_reserve_neg[92]: y_neg[92] <= 0
 c_reserve_neg[93]: y_neg[93] <= 0
 c_reserve_neg[94]: y_neg[94] <= 0
 c_reserve_neg[95]: y_neg[95] <= 0
 c_charge_re[0]: y_chg[0] <= 250
 c_charge_re[1]: y_chg[1] <= 250
 c_charge_re[2]: y_chg[2] <= 250
 c_charge_re[3]: y_chg[3] <= 250
 c_charge_re[4]: y_chg[4] <= 250
 c_charge_re[5]: y_chg[5] <= 250
 c_charge_re[6]: y_chg[6] <= 250
 c_charge_re[7]: y_chg[7] <= 250
 c_charge_re[8]: y_chg[8] <= 250
 c_charge_re[9]: y_chg[9] <= 250
 c_charge_re[10]: y_chg[10] <= 250
 c_charge_re[11]: y_chg[11] <= 250
 c_charge_re[12]: y_chg[12] <= 250
 c_charge_re[13]: y_chg[13] <= 250
 c_charge_re[14]: y_chg[14] <= 250
 c_charge_re[15]: y_chg[15] <= 250
 c_charge_re[16]: y_chg[16] <= 250
 c_charge_re[17]: y_chg[17] <= 250
 c_charge_re[18]: y_chg[18] <= 250
 c_charge_re[19]: y_chg[19] <= 250
 c_charge_re[20]: y_chg[20] <= 250
 c_charge_re[21]: y_chg[21] <= 250
 c_charge_re[22]: y_chg[22] <= 250
 c_charge_re[23]: y_chg[23] <= 250
 c_charge_re[24]: y_chg[24] <= 250
 c_charge_re[25]: y_chg[25] <= 250
 c_charge_re[26]: y_chg[26] <= 250
 c_charge_re[27]: y_chg[27] <= 250
 c_charge_re[28]: y_chg[28] <= 250
 c_charge_re[29]: y_chg[29] <= 250
 c_charge_re[30]: y_chg[30] <= 250
 c_charge_re[31]: y_chg[31] <= 250
 c_charge_re[32]: y_chg[32] <= 250
 c_charge_re[33]: y_chg[33] <= 250
 c_charge_re[34]: y_chg[34] <= 250
 c_charge_re[35]: y_chg[35] <= 250
 c_charge_re[36]: y_chg[36] <= 250
 c_charge_re[37]: y_chg[37] <= 250
 c_charge_re[38]: y_chg[38] <= 250
 c_charge_re[39]: y_chg[39] <= 250
 c_charge_re[40]: y_chg[40] <= 250
 c_charge_re[41]: y_chg[41] <= 250
 c_charge_re[42]: y_chg[42] <= 250
 c_charge_re[43]: y_chg[43] <= 250
 c_charge_re[44]: y_chg[44] <= 250
 c_charge_re[45]: y_chg[45] <= 250
 c_charge_re[46]: y_chg[46] <= 250
 c_charge_re[47]: y_chg[47] <= 250
 c_charge_re[48]: y_chg[48] <= 250
 c_charge_re[49]: y_chg[49] <= 250
 c_charge_re[50]: y_chg[50] <= 250
 c_charge_re[51]: y_chg[51] <= 250
 c_charge_re[52]: y_chg[52] <= 250
 c_charge_re[53]: y_chg[53] <= 250
 c_charge_re[54]: y_chg[54] <= 250
 c_charge_re[55]: y_chg[55] <= 250
 c_charge_re[56]: y_chg[56] <= 250
 c_charge_re[57]: y_chg[57] <= 250
 c_charge_re[58]: y_chg[58] <= 250
 c_charge_re[59]: y_chg[59] <= 250
 c_charge_re[60]: y_chg[60] <= 250
 c_charge_re[61]: y_chg[61] <= 250
 c_charge_re[62]: y_chg[62] <= 250
 c_charge_re[63]: y_chg[63] <= 250
 c_charge_re[64]: y_chg[64] <= 250
 c_charge_re[65]: y_chg[65] <= 250
 c_charge_re[66]: y_chg[66] <= 250
 c_charge_re[67]: y_chg[67] <= 250
 c_charge_re[68]: y_chg[68] <= 250
 c_charge_re[69]: y_chg[69] <= 250
 c_charge_re[70]: y_chg[70] <= 250
 c_charge_re[71]: y_chg[71] <= 250
 c_charge_re[72]: y_chg[72] <= 250
 c_charge_re[73]: y_chg[73] <= 250
 c_charge_re[74]: y_chg[74] <= 250
 c_charge_re[75]: y_chg[75] <= 250
 c_charge_re[76]: y_chg[76] <= 250
 c_charge_re[77]: y_chg[77] <= 250
 c_charge_re[78]: y_chg[78] <= 250
 c_charge_re[79]: y_chg[79] <= 250
 c_charge_re[80]: y_chg[80] <= 250
 c_charge_re[81]: y_chg[81] <= 250
 c_charge_re[82]: y_chg[82] <= 250
 c_charge_re[83]: y_chg[83] <= 250
 c_charge_re[84]: y_chg[84] <= 250
 c_charge_re[85]: y_chg[85] <= 250
 c_charge_re[86]: y_chg[86] <= 250
 c_charge_re[87]: y_chg[87] <= 250
 c_charge_re[88]: y_chg[88] <= 250
 c_charge_re[89]: y_chg[89] <= 250
 c_charge_re[90]: y_chg[90] <= 250
 c_charge_re[91]: y_chg[91] <= 250
 c_charge_re[92]: y_chg[92] <= 250
 c_charge_re[93]: y_chg[93] <= 250
 c_charge_re[94]: y_chg[94] <= 250
 c_charge_re[95]: y_chg[95] <= 250
 c_discharge_re[0]: y_dis[0] <= 250
 c_discharge_re[1]: y_dis[1] <= 250
 c_discharge_re[2]: y_dis[2] <= 250
 c_discharge_re[3]: y_dis[3] <= 250
 c_discharge_re[4]: y_dis[4] <= 250
 c_discharge_re[5]: y_dis[5] <= 250
 c_discharge_re[6]: y_dis[6] <= 250
 c_discharge_re[7]: y_dis[7] <= 250
 c_discharge_re[8]: y_dis[8] <= 250
 c_discharge_re[9]: y_dis[9] <= 250
 c_discharge_re[10]: y_dis[10] <= 250
 c_discharge_re[11]: y_dis[11] <= 250
 c_discharge_re[12]: y_dis[12] <= 250
 c_discharge_re[13]: y_dis[13] <= 250
 c_discharge_re[14]: y_dis[14] <= 250
 c_discharge_re[15]: y_dis[15] <= 250
 c_discharge_re[16]: y_dis[16] <= 250
 c_discharge_re[17]: y_dis[17] <= 250
 c_discharge_re[18]: y_dis[18] <= 250
 c_discharge_re[19]: y_dis[19] <= 250
 c_discharge_re[20]: y_dis[20] <= 250
 c_discharge_re[21]: y_dis[21] <= 250
 c_discharge_re[22]: y_dis[22] <= 250
 c_discharge_re[23]: y_dis[23] <= 250
 c_discharge_re[24]: y_dis[24] <= 250
 c_discharge_re[25]: y_dis[25] <= 250
 c_discharge_re[26]: y_dis[26] <= 250
 c_discharge_re[27]: y_dis[27] <= 250
 c_discharge_re[28]: y_dis[28] <= 250
 c_discharge_re[29]: y_dis[29] <= 250
 c_discharge_re[30]: y_dis[30] <= 250
 c_discharge_re[31]: y_dis[31] <= 250
 c_discharge_re[32]: y_dis[32] <= 250
 c_discharge_re[33]: y_dis[33] <= 250
 c_discharge_re[34]: y_dis[34] <= 250
 c_discharge_re[35]: y_dis[35] <= 250
 c_discharge_re[36]: y_dis[36] <= 250
 c_discharge_re[37]: y_dis[37] <= 250
 c_discharge_re[38]: y_dis[38] <= 250
 c_discharge_re[39]: y_dis[39] <= 250
 c_discharge_re[40]: y_dis[40] <= 250
 c_discharge_re[41]: y_dis[41] <= 250
 c_discharge_re[42]: y_dis[42] <= 250
 c_discharge_re[43]: y_dis[43] <= 250
 c_discharge_re[44]: y_dis[44] <= 250
 c_discharge_re[45]: y_dis[45] <= 250
 c_discharge_re[46]: y_dis[46] <= 250
 c_discharge_re[47]: y_dis[47] <= 250
 c_discharge_re[48]: y_dis[48] <= 250
 c_discharge_re[49]: y_dis[49] <= 250
 c_discharge_re[50]: y_dis[50] <= 250
 c_discharge_re[51]: y_dis[51] <= 250
 c_discharge_re[52]: y_dis[52] <= 250
 c_discharge_re[53]: y_dis[53] <= 250
 c_discharge_re[54]: y_dis[54] <= 250
 c_discharge_re[55]: y_dis[55] <= 250
 c_discharge_re[56]: y_dis[56] <= 250
 c_discharge_re[57]: y_dis[57] <= 250
 c_discharge_re[58]: y_dis[58] <= 250
 c_discharge_re[59]: y_dis[59] <= 250
 c_discharge_re[60]: y_dis[60] <= 250
 c_discharge_re[61]: y_dis[61] <= 250
 c_discharge_re[62]: y_dis[62] <= 250
 c_discharge_re[63]: y_dis[63] <= 250
 c_discharge_re[64]: y_dis[64] <= 250
 c_discharge_re[65]: y_dis[65] <= 250
 c_discharge_re[66]: y_dis[66] <= 250
 c_discharge_re[67]: y_dis[67] <= 250
 c_discharge_re[68]: y_dis[68] <= 250
 c_discharge_re[69]: y_dis[69] <= 250
 c_discharge_re[70]: y_dis[70] <= 250
 c_discharge_re[71]: y_dis[71] <= 250
 c_discharge_re[72]: y_dis[72] <= 250
 c_discharge_re[73]: y_dis[73] <= 250
 c_discharge_re[74]: y_dis[74] <= 250
 c_discharge_re[75]: y_dis[75] <= 250
 c_discharge_re[76]: y_dis[76] <= 250
 c_discharge_re[77]: y_dis[77] <= 250
 c_discharge_re[78]: y_dis[78] <= 250
 c_discharge_re[79]: y_dis[79] <= 250
 c_discharge_re[80]: y_dis[80] <= 250
 c_discharge_re[81]: y_dis[81] <= 250
 c_discharge_re[82]: y_dis[82] <= 250
 c_discharge_re[83]: y_dis[83] <= 250
 c_discharge_re[84]: y_dis[84] <= 250
 c_discharge_re[85]: y_dis[85] <= 250
 c_discharge_re[86]: y_dis[86] <= 250
 c_discharge_re[87]: y_dis[87] <= 250
 c_discharge_re[88]: y_dis[88] <= 250
 c_discharge_re[89]: y_dis[89] <= 250
 c_discharge_re[90]: y_dis[90] <= 250
 c_discharge_re[91]: y_dis[91] <= 250
 c_discharge_re[92]: y_dis[92] <= 250
 c_discharge_re[93]: y_dis[93] <= 250
 c_discharge_re[94]: y_dis[94] <= 250
 c_discharge_re[95]: y_dis[95] <= 250
 c_SOC_min[0]: - y_S[0] <= -113.4
 c_SOC_min[1]: - y_S[1] <= -113.4
 c_SOC_min[2]: - y_S[2] <= -113.4
 c_SOC_min[3]: - y_S[3] <= -113.4
 c_SOC_min[4]: - y_S[4] <= -113.4
 c_SOC_min[5]: - y_S[5] <= -113.4
 c_SOC_min[6]: - y_S[6] <= -113.4
 c_SOC_min[7]: - y_S[7] <= -113.4
 c_SOC_min[8]: - y_S[8] <= -113.4
 c_SOC_min[9]: - y_S[9] <= -113.4
 c_SOC_min[10]: - y_S[10] <= -113.4
 c_SOC_min[11]: - y_S[11] <= -113.4
 c_SOC_min[12]: - y_S[12] <= -113.4
 c_SOC_min[13]: - y_S[13] <= -113.4
 c_SOC_min[14]: - y_S[14] <= -113.4
 c_SOC_min[15]: - y_S[15] <= -113.4
 c_SOC_min[16]: - y_S[16] <= -113.4
 c_SOC_min[17]: - y_S[17] <= -113.4
 c_SOC_min[18]: - y_S[18] <= -113.4
 c_SOC_min[19]: - y_S[19] <= -113.4
 c_SOC_min[20]: - y_S[20] <= -113.4
 c_SOC_min[21]: - y_S[21] <= -113.4
 c_SOC_min[22]: - y_S[22] <= -113.4
 c_SOC_min[23]: - y_S[23] <= -113.4
 c_SOC_min[24]: - y_S[24] <= -113.4
 c_SOC_min[25]: - y_S[25] <= -113.4
 c_SOC_min[26]: - y_S[26] <= -113.4
 c_SOC_min[27]: - y_S[27] <= -113.4
 c_SOC_min[28]: - y_S[28] <= -113.4
 c_SOC_min[29]: - y_S[29] <= -113.4
 c_SOC_min[30]: - y_S[30] <= -113.4
 c_SOC_min[31]: - y_S[31] <= -113.4
 c_SOC_min[32]: - y_S[32] <= -113.4
 c_SOC_min[33]: - y_S[33] <= -113.4
 c_SOC_min[34]: - y_S[34] <= -113.4
 c_SOC_min[35]: - y_S[35] <= -113.4
 c_SOC_min[36]: - y_S[36] <= -113.4
 c_SOC_min[37]: - y_S[37] <= -113.4
 c_SOC_min[38]: - y_S[38] <= -113.4
 c_SOC_min[39]: - y_S[39] <= -113.4
 c_SOC_min[40]: - y_S[40] <= -113.4
 c_SOC_min[41]: - y_S[41] <= -113.4
 c_SOC_min[42]: - y_S[42] <= -113.4
 c_SOC_min[43]: - y_S[43] <= -113.4
 c_SOC_min[44]: - y_S[44] <= -113.4
 c_SOC_min[45]: - y_S[45] <= -113.4
 c_SOC_min[46]: - y_S[46] <= -113.4
 c_SOC_min[47]: - y_S[47] <= -113.4
 c_SOC_min[48]: - y_S[48] <= -113.4
 c_SOC_min[49]: - y_S[49] <= -113.4
 c_SOC_min[50]: - y_S[50] <= -113.4
 c_SOC_min[51]: - y_S[51] <= -113.4
 c_SOC_min[52]: - y_S[52] <= -113.4
 c_SOC_min[53]: - y_S[53] <= -113.4
 c_SOC_min[54]: - y_S[54] <= -113.4
 c_SOC_min[55]: - y_S[55] <= -113.4
 c_SOC_min[56]: - y_S[56] <= -113.4
 c_SOC_min[57]: - y_S[57] <= -113.4
 c_SOC_min[58]: - y_S[58] <= -113.4
 c_SOC_min[59]: - y_S[59] <= -113.4
 c_SOC_min[60]: - y_S[60] <= -113.4
 c_SOC_min[61]: - y_S[61] <= -113.4
 c_SOC_min[62]: - y_S[62] <= -113.4
 c_SOC_min[63]: - y_S[63] <= -113.4
 c_SOC_min[64]: - y_S[64] <= -113.4
 c_SOC_min[65]: - y_S[65] <= -113.4
 c_SOC_min[66]: - y_S[66] <= -113.4
 c_SOC_min[67]: - y_S[67] <= -113.4
 c_SOC_min[68]: - y_S[68] <= -113.4
 c_SOC_min[69]: - y_S[69] <= -113.4
 c_SOC_min[70]: - y_S[70] <= -113.4
 c_SOC_min[71]: - y_S[71] <= -113.4
 c_SOC_min[72]: - y_S[72] <= -113.4
 c_SOC_min[73]: - y_S[73] <= -113.4
 c_SOC_min[74]: - y_S[74] <= -113.4
 c_SOC_min[75]: - y_S[75] <= -113.4
 c_SOC_min[76]: - y_S[76] <= -113.4
 c_SOC_min[77]: - y_S[77] <= -113.4
 c_SOC_min[78]: - y_S[78] <= -113.4
 c_SOC_min[79]: - y_S[79] <= -113.4
 c_SOC_min[80]: - y_S[80] <= -113.4
 c_SOC_min[81]: - y_S[81] <= -113.4
 c_SOC_min[82]: - y_S[82] <= -113.4
 c_SOC_min[83]: - y_S[83] <= -113.4
 c_SOC_min[84]: - y_S[84] <= -113.4
 c_SOC_min[85]: - y_S[85] <= -113.4
 c_SOC_min[86]: - y_S[86] <= -113.4
 c_SOC_min[87]: - y_S[87] <= -113.4
 c_SOC_min[88]: - y_S[88] <= -113.4
 c_SOC_min[89]: - y_S[89] <= -113.4
 c_SOC_min[90]: - y_S[90] <= -113.4
 c_SOC_min[91]: - y_S[91] <= -113.4
 c_SOC_min[92]: - y_S[92] <= -113.4
 c_SOC_min[93]: - y_S[93] <= -113.4
 c_SOC_min[94]: - y_S[94] <= -113.4
 c_SOC_min[95]: - y_S[95] <= -113.4
 c_S_max[0]: y_S[0] <= 453.6
 c_S_max[1]: y_S[1] <= 453.6
 c_S_max[2]: y_S[2] <= 453.6
 c_S_max[3]: y_S[3] <= 453.6
 c_S_max[4]: y_S[4] <= 453.6
 c_S_max[5]: y_S[5] <= 453.6
 c_S_max[6]: y_S[6] <= 453.6
 c_S_max[7]: y_S[7] <= 453.6
 c_S_max[8]: y_S[8] <= 453.6
 c_S_max[9]: y_S[9] <= 453.6
 c_S_max[10]: y_S[10] <= 453.6
 c_S_max[11]: y_S[11] <= 453.6
 c_S_max[12]: y_S[12] <= 453.6
 c_S_max[13]: y_S[13] <= 453.6
 c_S_max[14]: y_S[14] <= 453.6
 c_S_max[15]: y_S[15] <= 453.6
 c_S_max[16]: y_S[16] <= 453.6
 c_S_max[17]: y_S[17] <= 453.6
 c_S_max[18]: y_S[18] <= 453.6
 c_S_max[19]: y_S[19] <= 453.6
 c_S_max[20]: y_S[20] <= 453.6
 c_S_max[21]: y_S[21] <= 453.6
 c_S_max[22]: y_S[22] <= 453.6
 c_S_max[23]: y_S[23] <= 453.6
 c_S_max[24]: y_S[24] <= 453.6
 c_S_max[25]: y_S[25] <= 453.6
 c_S_max[26]: y_S[26] <= 453.6
 c_S_max[27]: y_S[27] <= 453.6
 c_S_max[28]: y_S[28] <= 453.6
 c_S_max[29]: y_S[29] <= 453.6
 c_S_max[30]: y_S[30] <= 453.6
 c_S_max[31]: y_S[31] <= 453.6
 c_S_max[32]: y_S[32] <= 453.6
 c_S_max[33]: y_S[33] <= 453.6
 c_S_max[34]: y_S[34] <= 453.6
 c_S_max[35]: y_S[35] <= 453.6
 c_S_max[36]: y_S[36] <= 453.6
 c_S_max[37]: y_S[37] <= 453.6
 c_S_max[38]: y_S[38] <= 453.6
 c_S_max[39]: y_S[39] <= 453.6
 c_S_max[40]: y_S[40] <= 453.6
 c_S_max[41]: y_S[41] <= 453.6
 c_S_max[42]: y_S[42] <= 453.6
 c_S_max[43]: y_S[43] <= 453.6
 c_S_max[44]: y_S[44] <= 453.6
 c_S_max[45]: y_S[45] <= 453.6
 c_S_max[46]: y_S[46] <= 453.6
 c_S_max[47]: y_S[47] <= 453.6
 c_S_max[48]: y_S[48] <= 453.6
 c_S_max[49]: y_S[49] <= 453.6
 c_S_max[50]: y_S[50] <= 453.6
 c_S_max[51]: y_S[51] <= 453.6
 c_S_max[52]: y_S[52] <= 453.6
 c_S_max[53]: y_S[53] <= 453.6
 c_S_max[54]: y_S[54] <= 453.6
 c_S_max[55]: y_S[55] <= 453.6
 c_S_max[56]: y_S[56] <= 453.6
 c_S_max[57]: y_S[57] <= 453.6
 c_S_max[58]: y_S[58] <= 453.6
 c_S_max[59]: y_S[59] <= 453.6
 c_S_max[60]: y_S[60] <= 453.6
 c_S_max[61]: y_S[61] <= 453.6
 c_S_max[62]: y_S[62] <= 453.6
 c_S_max[63]: y_S[63] <= 453.6
 c_S_max[64]: y_S[64] <= 453.6
 c_S_max[65]: y_S[65] <= 453.6
 c_S_max[66]: y_S[66] <= 453.6
 c_S_max[67]: y_S[67] <= 453.6
 c_S_max[68]: y_S[68] <= 453.6
 c_S_max[69]: y_S[69] <= 453.6
 c_S_max[70]: y_S[70] <= 453.6
 c_S_max[71]: y_S[71] <= 453.6
 c_S_max[72]: y_S[72] <= 453.6
 c_S_max[73]: y_S[73] <= 453.6
 c_S_max[74]: y_S[74] <= 453.6
 c_S_max[75]: y_S[75] <= 453.6
 c_S_max[76]: y_S[76] <= 453.6
 c_S_max[77]: y_S[77] <= 453.6
 c_S_max[78]: y_S[78] <= 453.6
 c_S_max[79]: y_S[79] <= 453.6
 c_S_max[80]: y_S[80] <= 453.6
 c_S_max[81]: y_S[81] <= 453.6
 c_S_max[82]: y_S[82] <= 453.6
 c_S_max[83]: y_S[83] <= 453.6
 c_S_max[84]: y_S[84] <= 453.6
 c_S_max[85]: y_S[85] <= 453.6
 c_S_max[86]: y_S[86] <= 453.6
 c_S_max[87]: y_S[87] <= 453.6
 c_S_max[88]: y_S[88] <= 453.6
 c_S_max[89]: y_S[89] <= 453.6
 c_S_max[90]: y_S[90] <= 453.6
 c_S_max[91]: y_S[91] <= 453.6
 c_S_max[92]: y_S[92] <= 453.6
 c_S_max[93]: y_S[93] <= 453.6
 c_S_max[94]: y_S[94] <= 453.6
 c_S_max[95]: y_S[95] <= 453.6
 c_ESS_first_period: - 0.2325 y_chg[0] + 0.2688172043010753 y_dis[0]
   + y_S[0] = 187.5
 c_SOC_re[1]: - 0.2325 y_chg[1] + 0.2688172043010753 y_dis[1] - y_S[0]
   + y_S[1] = 0
 c_SOC_re[2]: - 0.2325 y_chg[2] + 0.2688172043010753 y_dis[2] - y_S[1]
   + y_S[2] = 0
 c_SOC_re[3]: - 0.2325 y_chg[3] + 0.2688172043010753 y_dis[3] - y_S[2]
   + y_S[3] = 0
 c_SOC_re[4]: - 0.2325 y_chg[4] + 0.2688172043010753 y_dis[4] - y_S[3]
   + y_S[4] = 0
 c_SOC_re[5]: - 0.2325 y_chg[5] + 0.2688172043010753 y_dis[5] - y_S[4]
   + y_S[5] = 0
 c_SOC_re[6]: - 0.2325 y_chg[6] + 0.2688172043010753 y_dis[6] - y_S[5]
   + y_S[6] = 0
 c_SOC_re[7]: - 0.2325 y_chg[7] + 0.2688172043010753 y_dis[7] - y_S[6]
   + y_S[7] = 0
 c_SOC_re[8]: - 0.2325 y_chg[8] + 0.2688172043010753 y_dis[8] - y_S[7]
   + y_S[8] = 0
 c_SOC_re[9]: - 0.2325 y_chg[9] + 0.2688172043010753 y_dis[9] - y_S[8]
   + y_S[9] = 0
 c_SOC_re[10]: - 0.2325 y_chg[10] + 0.2688172043010753 y_dis[10] - y_S[9]
   + y_S[10] = 0
 c_SOC_re[11]: - 0.2325 y_chg[11] + 0.2688172043010753 y_dis[11] - y_S[10]
   + y_S[11] = 0
 c_SOC_re[12]: - 0.2325 y_chg[12] + 0.2688172043010753 y_dis[12] - y_S[11]
   + y_S[12] = 0
 c_SOC_re[13]: - 0.2325 y_chg[13] + 0.2688172043010753 y_dis[13] - y_S[12]
   + y_S[13] = 0
 c_SOC_re[14]: - 0.2325 y_chg[14] + 0.2688172043010753 y_dis[14] - y_S[13]
   + y_S[14] = 0
 c_SOC_re[15]: - 0.2325 y_chg[15] + 0.2688172043010753 y_dis[15] - y_S[14]
   + y_S[15] = 0
 c_SOC_re[16]: - 0.2325 y_chg[16] + 0.2688172043010753 y_dis[16] - y_S[15]
   + y_S[16] = 0
 c_SOC_re[17]: - 0.2325 y_chg[17] + 0.2688172043010753 y_dis[17] - y_S[16]
   + y_S[17] = 0
 c_SOC_re[18]: - 0.2325 y_chg[18] + 0.2688172043010753 y_dis[18] - y_S[17]
   + y_S[18] = 0
 c_SOC_re[19]: - 0.2325 y_chg[19] + 0.2688172043010753 y_dis[19] - y_S[18]
   + y_S[19] = 0
 c_SOC_re[20]: - 0.2325 y_chg[20] + 0.2688172043010753 y_dis[20] - y_S[19]
   + y_S[20] = 0
 c_SOC_re[21]: - 0.2325 y_chg[21] + 0.2688172043010753 y_dis[21] - y_S[20]
   + y_S[21] = 0
 c_SOC_re[22]: - 0.2325 y_chg[22] + 0.2688172043010753 y_dis[22] - y_S[21]
   + y_S[22] = 0
 c_SOC_re[23]: - 0.2325 y_chg[23] + 0.2688172043010753 y_dis[23] - y_S[22]
   + y_S[23] = -3.712055459586395
 c_SOC_re[24]: - 0.2325 y_chg[24] + 0.2688172043010753 y_dis[24] - y_S[23]
   + y_S[24] = 0
 c_SOC_re[25]: - 0.2325 y_chg[25] + 0.2688172043010753 y_dis[25] - y_S[24]
   + y_S[25] = 0
 c_SOC_re[26]: - 0.2325 y_chg[26] + 0.2688172043010753 y_dis[26] - y_S[25]
   + y_S[26] = -8.741709622003698
 c_SOC_re[27]: - 0.2325 y_chg[27] + 0.2688172043010753 y_dis[27] - y_S[26]
   + y_S[27] = 0
 c_SOC_re[28]: - 0.2325 y_chg[28] + 0.2688172043010753 y_dis[28] - y_S[27]
   + y_S[28] = -4.946234918409778
 c_SOC_re[29]: - 0.2325 y_chg[29] + 0.2688172043010753 y_dis[29] - y_S[28]
   + y_S[29] = 0
 c_SOC_re[30]: - 0.2325 y_chg[30] + 0.2688172043010753 y_dis[30] - y_S[29]
   + y_S[30] = 0
 c_SOC_re[31]: - 0.2325 y_chg[31] + 0.2688172043010753 y_dis[31] - y_S[30]
   + y_S[31] = 0
 c_SOC_re[32]: - 0.2325 y_chg[32] + 0.2688172043010753 y_dis[32] - y_S[31]
   + y_S[32] = 0
 c_SOC_re[33]: - 0.2325 y_chg[33] + 0.2688172043010753 y_dis[33] - y_S[32]
   + y_S[33] = 0
 c_SOC_re[34]: - 0.2325 y_chg[34] + 0.2688172043010753 y_dis[34] - y_S[33]
   + y_S[34] = 0
 c_SOC_re[35]: - 0.2325 y_chg[35] + 0.2688172043010753 y_dis[35] - y_S[34]
   + y_S[35] = 0
 c_SOC_re[36]: - 0.2325 y_chg[36] + 0.2688172043010753 y_dis[36] - y_S[35]
   + y_S[36] = 0
 c_SOC_re[37]: - 0.2325 y_chg[37] + 0.2688172043010753 y_dis[37] - y_S[36]
   + y_S[37] = 0
 c_SOC_re[38]: - 0.2325 y_chg[38] + 0.2688172043010753 y_dis[38] - y_S[37]
   + y_S[38] = 1.172090776089679
 c_SOC_re[39]: - 0.2325 y_chg[39] + 0.2688172043010753 y_dis[39] - y_S[38]
   + y_S[39] = -1.172090776089675
 c_SOC_re[40]: - 0.2325 y_chg[40] + 0.2688172043010753 y_dis[40] - y_S[39]
   + y_S[40] = 3.023965458726407
 c_SOC_re[41]: - 0.2325 y_chg[41] + 0.2688172043010753 y_dis[41] - y_S[40]
   + y_S[41] = 5.333663636669817
 c_SOC_re[42]: - 0.2325 y_chg[42] + 0.2688172043010753 y_dis[42] - y_S[41]
   + y_S[42] = -8.357629095396216
 c_SOC_re[43]: - 0.2325 y_chg[43] + 0.2688172043010753 y_dis[43] - y_S[42]
   + y_S[43] = 3.759672001365675
 c_SOC_re[44]: - 0.2325 y_chg[44] + 0.2688172043010753 y_dis[44] - y_S[43]
   + y_S[44] = 4.903138963589316
 c_SOC_re[45]: - 0.2325 y_chg[45] + 0.2688172043010753 y_dis[45] - y_S[44]
   + y_S[45] = 6.491755708905966
 c_SOC_re[46]: - 0.2325 y_chg[46] + 0.2688172043010753 y_dis[46] - y_S[45]
   + y_S[46] = 14.75043950053701
 c_SOC_re[47]: - 0.2325 y_chg[47] + 0.2688172043010753 y_dis[47] - y_S[46]
   + y_S[47] = 16.00061836179101
 c_SOC_re[48]: - 0.2325 y_chg[48] + 0.2688172043010753 y_dis[48] - y_S[47]
   + y_S[48] = 17.2450159254431
 c_SOC_re[49]: - 0.2325 y_chg[49] + 0.2688172043010753 y_dis[49] - y_S[48]
   + y_S[49] = 20.32499026062479
 c_SOC_re[50]: - 0.2325 y_chg[50] + 0.2688172043010753 y_dis[50] - y_S[49]
   + y_S[50] = 17.74620980639808
 c_SOC_re[51]: - 0.2325 y_chg[51] + 0.2688172043010753 y_dis[51] - y_S[50]
   + y_S[51] = 17.98319568384721
 c_SOC_re[52]: - 0.2325 y_chg[52] + 0.2688172043010753 y_dis[52] - y_S[51]
   + y_S[52] = 18.21760599924194
 c_SOC_re[53]: - 0.2325 y_chg[53] + 0.2688172043010753 y_dis[53] - y_S[52]
   + y_S[53] = 17.55267652436472
 c_SOC_re[54]: - 0.2325 y_chg[54] + 0.2688172043010753 y_dis[54] - y_S[53]
   + y_S[54] = 18.5454033786408
 c_SOC_re[55]: - 0.2325 y_chg[55] + 0.2688172043010753 y_dis[55] - y_S[54]
   + y_S[55] = 15.54728011875985
 c_SOC_re[56]: - 0.2325 y_chg[56] + 0.2688172043010753 y_dis[56] - y_S[55]
   + y_S[56] = 6.224632994117372
 c_SOC_re[57]: - 0.2325 y_chg[57] + 0.2688172043010753 y_dis[57] - y_S[56]
   + y_S[57] = 6.871875660331226
 c_SOC_re[58]: - 0.2325 y_chg[58] + 0.2688172043010753 y_dis[58] - y_S[57]
   + y_S[58] = 6.594808503048992
 c_SOC_re[59]: - 0.2325 y_chg[59] + 0.2688172043010753 y_dis[59] - y_S[58]
   + y_S[59] = 9.189662874880659
 c_SOC_re[60]: - 0.2325 y_chg[60] + 0.2688172043010753 y_dis[60] - y_S[59]
   + y_S[60] = 6.136034541460631
 c_SOC_re[61]: - 0.2325 y_chg[61] + 0.2688172043010753 y_dis[61] - y_S[60]
   + y_S[61] = 2.714983192651651
 c_SOC_re[62]: - 0.2325 y_chg[62] + 0.2688172043010753 y_dis[62] - y_S[61]
   + y_S[62] = -7.181370974562422
 c_SOC_re[63]: - 0.2325 y_chg[63] + 0.2688172043010753 y_dis[63] - y_S[62]
   + y_S[63] = 7.181370974562412
 c_SOC_re[64]: - 0.2325 y_chg[64] + 0.2688172043010753 y_dis[64] - y_S[63]
   + y_S[64] = -1.698923166986444
 c_SOC_re[65]: - 0.2325 y_chg[65] + 0.2688172043010753 y_dis[65] - y_S[64]
   + y_S[65] = 1.698923166986436
 c_SOC_re[66]: - 0.2325 y_chg[66] + 0.2688172043010753 y_dis[66] - y_S[65]
   + y_S[66] = -10.99783032859454
 c_SOC_re[67]: - 0.2325 y_chg[67] + 0.2688172043010753 y_dis[67] - y_S[66]
   + y_S[67] = 10.99783032859453
 c_SOC_re[68]: - 0.2325 y_chg[68] + 0.2688172043010753 y_dis[68] - y_S[67]
   + y_S[68] = 0
 c_SOC_re[69]: - 0.2325 y_chg[69] + 0.2688172043010753 y_dis[69] - y_S[68]
   + y_S[69] = 0
 c_SOC_re[70]: - 0.2325 y_chg[70] + 0.2688172043010753 y_dis[70] - y_S[69]
   + y_S[70] = -12.32189465594548
 c_SOC_re[71]: - 0.2325 y_chg[71] + 0.2688172043010753 y_dis[71] - y_S[70]
   + y_S[71] = 0
 c_SOC_re[72]: - 0.2325 y_chg[72] + 0.2688172043010753 y_dis[72] - y_S[71]
   + y_S[72] = 0
 c_SOC_re[73]: - 0.2325 y_chg[73] + 0.2688172043010753 y_dis[73] - y_S[72]
   + y_S[73] = 0
 c_SOC_re[74]: - 0.2325 y_chg[74] + 0.2688172043010753 y_dis[74] - y_S[73]
   + y_S[74] = -1.636138526342248
 c_SOC_re[75]: - 0.2325 y_chg[75] + 0.2688172043010753 y_dis[75] - y_S[74]
   + y_S[75] = -6.34772777557373
 c_SOC_re[76]: - 0.2325 y_chg[76] + 0.2688172043010753 y_dis[76] - y_S[75]
   + y_S[76] = -7.874524849717335
 c_SOC_re[77]: - 0.2325 y_chg[77] + 0.2688172043010753 y_dis[77] - y_S[76]
   + y_S[77] = -9.056724656012751
 c_SOC_re[78]: - 0.2325 y_chg[78] + 0.2688172043010753 y_dis[78] - y_S[77]
   + y_S[78] = -8.838983376820881
 c_SOC_re[79]: - 0.2325 y_chg[79] + 0.2688172043010753 y_dis[79] - y_S[78]
   + y_S[79] = -8.941135227039295
 c_SOC_re[80]: - 0.2325 y_chg[80] + 0.2688172043010753 y_dis[80] - y_S[79]
   + y_S[80] = -8.683345394749795
 c_SOC_re[81]: - 0.2325 y_chg[81] + 0.2688172043010753 y_dis[81] - y_S[80]
   + y_S[81] = -8.565065168565319
 c_SOC_re[82]: - 0.2325 y_chg[82] + 0.2688172043010753 y_dis[82] - y_S[81]
   + y_S[82] = -8.850010748832457
 c_SOC_re[83]: - 0.2325 y_chg[83] + 0.2688172043010753 y_dis[83] - y_S[82]
   + y_S[83] = -8.48979667950702
 c_SOC_re[84]: - 0.2325 y_chg[84] + 0.2688172043010753 y_dis[84] - y_S[83]
   + y_S[84] = -8.656734420407203
 c_SOC_re[85]: - 0.2325 y_chg[85] + 0.2688172043010753 y_dis[85] - y_S[84]
   + y_S[85] = -8.538454194222727
 c_SOC_re[86]: - 0.2325 y_chg[86] + 0.2688172043010753 y_dis[86] - y_S[85]
   + y_S[86] = -8.500824051518594
 c_SOC_re[87]: - 0.2325 y_chg[87] + 0.2688172043010753 y_dis[87] - y_S[86]
   + y_S[87] = -8.745440488220542
 c_SOC_re[88]: - 0.2325 y_chg[88] + 0.2688172043010753 y_dis[88] - y_S[87]
   + y_S[88] = -8.460767294770928
 c_SOC_re[89]: - 0.2325 y_chg[89] + 0.2688172043010753 y_dis[89] - y_S[88]
   + y_S[89] = -8.694645153578891
 c_SOC_re[90]: - 0.2325 y_chg[90] + 0.2688172043010753 y_dis[90] - y_S[89]
   + y_S[90] = -8.78335122139223
 c_SOC_re[91]: - 0.2325 y_chg[91] + 0.2688172043010753 y_dis[91] - y_S[90]
   + y_S[91] = -8.640878431258663
 c_SOC_re[92]: - 0.2325 y_chg[92] + 0.2688172043010753 y_dis[92] - y_S[91]
   + y_S[92] = -9.079324173670942
 c_SOC_re[93]: - 0.2325 y_chg[93] + 0.2688172043010753 y_dis[93] - y_S[92]
   + y_S[93] = -8.861582894479074
 c_SOC_re[94]: - 0.2325 y_chg[94] + 0.2688172043010753 y_dis[94] - y_S[93]
   + y_S[94] = -9.194913602644396
 c_SOC_re[95]: - 0.2325 y_chg[95] + 0.2688172043010753 y_dis[95] - y_S[94]
   + y_S[95] = -23.63773701472959
 c_ESS_last_period: y_S[95] = 187.5
 c_y_PV[0]: y_PV[0] = 1.690000057220459
 c_y_PV[1]: y_PV[1] = 1.690000057220459
 c_y_PV[2]: y_PV[2] = 1.700000047683716
 c_y_PV[3]: y_PV[3] = 1.700000047683716
 c_y_PV[4]: y_PV[4] = 1.700000047683716
 c_y_PV[5]: y_PV[5] = 2.039999961853027
 c_y_PV[6]: y_PV[6] = 2.039999961853027
 c_y_PV[7]: y_PV[7] = 2.039999961853027
 c_y_PV[8]: y_PV[8] = 2.039999961853027
 c_y_PV[9]: y_PV[9] = 2.039999961853027
 c_y_PV[10]: y_PV[10] = 2.380000114440918
 c_y_PV[11]: y_PV[11] = 2.380000114440918
 c_y_PV[12]: y_PV[12] = 2.369999885559082
 c_y_PV[13]: y_PV[13] = 2.369999885559082
 c_y_PV[14]: y_PV[14] = 2.369999885559082
 c_y_PV[15]: y_PV[15] = 2.710000038146973
 c_y_PV[16]: y_PV[16] = 2.710000038146973
 c_y_PV[17]: y_PV[17] = 2.710000038146973
 c_y_PV[18]: y_PV[18] = 2.710000038146973
 c_y_PV[19]: y_PV[19] = 2.710000038146973
 c_y_PV[20]: y_PV[20] = 3.049999952316284
 c_y_PV[21]: y_PV[21] = 3.049999952316284
 c_y_PV[22]: y_PV[22] = 3.049999952316284
 c_y_PV[23]: y_PV[23] = 3.049999952316284
 c_y_PV[24]: y_PV[24] = 3.049999952316284
 c_y_PV[25]: y_PV[25] = 3.390000104904175
 c_y_PV[26]: y_PV[26] = 3.049999952316284
 c_y_PV[27]: y_PV[27] = 11.19999980926514
 c_y_PV[28]: y_PV[28] = 20.25
 c_y_PV[29]: y_PV[29] = 32.2599983215332
 c_y_PV[30]: y_PV[30] = 24.03000068664551
 c_y_PV[31]: y_PV[31] = 61.09000015258789
 c_y_PV[32]: y_PV[32] = 100.4300003051758
 c_y_PV[33]: y_PV[33] = 134.8500061035156
 c_y_PV[34]: y_PV[34] = 154.7200012207031
 c_y_PV[35]: y_PV[35] = 175.3000030517578
 c_y_PV[36]: y_PV[36] = 219.9299926757813
 c_y_PV[37]: y_PV[37] = 254.6499938964844
 c_y_PV[38]: y_PV[38] = 279.2999877929688
 c_y_PV[39]: y_PV[39] = 292.739990234375
 c_y_PV[40]: y_PV[40] = 340.510009765625
 c_y_PV[41]: y_PV[41] = 361.7200012207031
 c_y_PV[42]: y_PV[42] = 297.3099975585938
 c_y_PV[43]: y_PV[43] = 351.8599853515625
 c_y_PV[44]: y_PV[44] = 416.6700134277344
 c_y_PV[45]: y_PV[45] = 429.7099914550781
 c_y_PV[46]: y_PV[46] = 442.3500061035156
 c_y_PV[47]: y_PV[47] = 449.0599975585938
 c_y_PV[48]: y_PV[48] = 461.3999938964844
 c_y_PV[49]: y_PV[49] = 488.760009765625
 c_y_PV[50]: y_PV[50] = 463.4200134277344
 c_y_PV[51]: y_PV[51] = 470.25
 c_y_PV[52]: y_PV[52] = 469.0899963378906
 c_y_PV[53]: y_PV[53] = 464.6300048828125
 c_y_PV[54]: y_PV[54] = 473.5499877929688
 c_y_PV[55]: y_PV[55] = 460.1700134277344
 c_y_PV[56]: y_PV[56] = 423.1900024414063
 c_y_PV[57]: y_PV[57] = 434.9599914550781
 c_y_PV[58]: y_PV[58] = 427.6499938964844
 c_y_PV[59]: y_PV[59] = 459.2999877929688
 c_y_PV[60]: y_PV[60] = 420.9599914550781
 c_y_PV[61]: y_PV[61] = 391.8999938964844
 c_y_PV[62]: y_PV[62] = 353.9599914550781
 c_y_PV[63]: y_PV[63] = 398.3500061035156
 c_y_PV[64]: y_PV[64] = 299.9400024414063
 c_y_PV[65]: y_PV[65] = 250.5099945068359
 c_y_PV[66]: y_PV[66] = 142.1000061035156
 c_y_PV[67]: y_PV[67] = 242.0800018310547
 c_y_PV[68]: y_PV[68] = 106.9800033569336
 c_y_PV[69]: y_PV[69] = 119.4499969482422
 c_y_PV[70]: y_PV[70] = 54.81000137329102
 c_y_PV[71]: y_PV[71] = 60.86000061035156
 c_y_PV[72]: y_PV[72] = 64.44999694824219
 c_y_PV[73]: y_PV[73] = 61.13999938964844
 c_y_PV[74]: y_PV[74] = 31.30999946594238
 c_y_PV[75]: y_PV[75] = 12.14999961853027
 c_y_PV[76]: y_PV[76] = 4.710000038146973
 c_y_PV[77]: y_PV[77] = 0.3600000143051147
 c_y_PV[78]: y_PV[78] = 0.3600000143051147
 c_y_PV[79]: y_PV[79] = 0.3600000143051147
 c_y_PV[80]: y_PV[80] = 0.7200000286102295
 c_y_PV[81]: y_PV[81] = 0.7200000286102295
 c_y_PV[82]: y_PV[82] = 0.6299999952316284
 c_y_PV[83]: y_PV[83] = 0.6299999952316284
 c_y_PV[84]: y_PV[84] = 1.070000052452087
 c_y_PV[85]: y_PV[85] = 1.070000052452087
 c_y_PV[86]: y_PV[86] = 1.070000052452087
 c_y_PV[87]: y_PV[87] = 1.070000052452087
 c_y_PV[88]: y_PV[88] = 1.399999976158142
 c_y_PV[89]: y_PV[89] = 1.320000052452087
 c_y_PV[90]: y_PV[90] = 1.399999976158142
 c_y_PV[91]: y_PV[91] = 1.399999976158142
 c_y_PV[92]: y_PV[92] = 1.720000028610229
 c_y_PV[93]: y_PV[93] = 1.669999957084656
 c_y_PV[94]: y_PV[94] = 1.720000028610229
 c_y_PV[95]: y_PV[95] = 1.720000028610229
 c_y_load[0]: y_load[0] = 274.239990234375
 c_y_load[1]: y_load[1] = 272.7000122070313
 c_y_load[2]: y_load[2] = 275.0499877929688
 c_y_load[3]: y_load[3] = 273.9299926757813
 c_y_load[4]: y_load[4] = 273.7799987792969
 c_y_load[5]: y_load[5] = 275.7900085449219
 c_y_load[6]: y_load[6] = 272.2000122070313
 c_y_load[7]: y_load[7] = 275.3599853515625
 c_y_load[8]: y_load[8] = 273.6400146484375
 c_y_load[9]: y_load[9] = 274.6199951171875
 c_y_load[10]: y_load[10] = 275.6499938964844
 c_y_load[11]: y_load[11] = 274.9200134277344
 c_y_load[12]: y_load[12] = 274.4800109863281
 c_y_load[13]: y_load[13] = 276.8200073242188
 c_y_load[14]: y_load[14] = 273.6700134277344
 c_y_load[15]: y_load[15] = 277.0799865722656
 c_y_load[16]: y_load[16] = 273.2799987792969
 c_y_load[17]: y_load[17] = 275.0299987792969
 c_y_load[18]: y_load[18] = 274.2999877929688
 c_y_load[19]: y_load[19] = 273.8299865722656
 c_y_load[20]: y_load[20] = 275.3699951171875
 c_y_load[21]: y_load[21] = 274.1799926757813
 c_y_load[22]: y_load[22] = 274.6900024414063
 c_y_load[23]: y_load[23] = 275.6300048828125
 c_y_load[24]: y_load[24] = 273.3599853515625
 c_y_load[25]: y_load[25] = 275.5
 c_y_load[26]: y_load[26] = 272.5
 c_y_load[27]: y_load[27] = 272.6300048828125
 c_y_load[28]: y_load[28] = 271.2099914550781
 c_y_load[29]: y_load[29] = 268.1700134277344
 c_y_load[30]: y_load[30] = 269.3999938964844
 c_y_load[31]: y_load[31] = 267.2000122070313
 c_y_load[32]: y_load[32] = 254.9499969482422
 c_y_load[33]: y_load[33] = 257.7000122070313
 c_y_load[34]: y_load[34] = 256.2000122070313
 c_y_load[35]: y_load[35] = 257.2900085449219
 c_y_load[36]: y_load[36] = 239.6000061035156
 c_y_load[37]: y_load[37] = 240.8000030517578
 c_y_load[38]: y_load[38] = 240.2799987792969
 c_y_load[39]: y_load[39] = 239.5599975585938
 c_y_load[40]: y_load[40] = 307.25
 c_y_load[41]: y_load[41] = 307.9200134277344
 c_y_load[42]: y_load[42] = 305.1799926757813
 c_y_load[43]: y_load[43] = 308.0599975585938
 c_y_load[44]: y_load[44] = 307.3900146484375
 c_y_load[45]: y_load[45] = 306.1400146484375
 c_y_load[46]: y_load[46] = 307.7300109863281
 c_y_load[47]: y_load[47] = 305.7099914550781
 c_y_load[48]: y_load[48] = 306.5299987792969
 c_y_load[49]: y_load[49] = 306.9599914550781
 c_y_load[50]: y_load[50] = 305.3800048828125
 c_y_load[51]: y_load[51] = 307.7799987792969
 c_y_load[52]: y_load[52] = 238.5099945068359
 c_y_load[53]: y_load[53] = 239
 c_y_load[54]: y_load[54] = 239.1499938964844
 c_y_load[55]: y_load[55] = 238.3200073242188
 c_y_load[56]: y_load[56] = 306.4299926757813
 c_y_load[57]: y_load[57] = 308.739990234375
 c_y_load[58]: y_load[58] = 306.7699890136719
 c_y_load[59]: y_load[59] = 236.8000030517578
 c_y_load[60]: y_load[60] = 305.4700012207031
 c_y_load[61]: y_load[61] = 307.6300048828125
 c_y_load[62]: y_load[62] = 305.5700073242188
 c_y_load[63]: y_load[63] = 306.5299987792969
 c_y_load[64]: y_load[64] = 306.2900085449219
 c_y_load[65]: y_load[65] = 304.2200012207031
 c_y_load[66]: y_load[66] = 303.0199890136719
 c_y_load[67]: y_load[67] = 234.3399963378906
 c_y_load[68]: y_load[68] = 303.3099975585938
 c_y_load[69]: y_load[69] = 236.8899993896484
 c_y_load[70]: y_load[70] = 236.5200042724609
 c_y_load[71]: y_load[71] = 236.3699951171875
 c_y_load[72]: y_load[72] = 245.6499938964844
 c_y_load[73]: y_load[73] = 246.9199981689453
 c_y_load[74]: y_load[74] = 244.6499938964844
 c_y_load[75]: y_load[75] = 244.5700073242188
 c_y_load[76]: y_load[76] = 253.0399932861328
 c_y_load[77]: y_load[77] = 253.3200073242188
 c_y_load[78]: y_load[78] = 252.6399993896484
 c_y_load[79]: y_load[79] = 252.9600067138672
 c_y_load[80]: y_load[80] = 259.8099975585938
 c_y_load[81]: y_load[81] = 259.4299926757813
 c_y_load[82]: y_load[82] = 260.3399963378906
 c_y_load[83]: y_load[83] = 259.1900024414063
 c_y_load[84]: y_load[84] = 265.0299987792969
 c_y_load[85]: y_load[85] = 264.6499938964844
 c_y_load[86]: y_load[86] = 264.5199890136719
 c_y_load[87]: y_load[87] = 265.3200073242188
 c_y_load[88]: y_load[88] = 270.3200073242188
 c_y_load[89]: y_load[89] = 271.0899963378906
 c_y_load[90]: y_load[90] = 271.3900146484375
 c_y_load[91]: y_load[91] = 270.9100036621094
 c_y_load[92]: y_load[92] = 275.3599853515625
 c_y_load[93]: y_load[93] = 274.6300048828125
 c_y_load[94]: y_load[94] = 275.75
 c_y_load[95]: y_load[95] = 275.8399963378906
 c_y_cut[0]: 0.1690000057220459 simga_cut_n[0,1]
   + 0.3380000114440918 simga_cut_n[0,2]
   + 0.5070000171661377 simga_cut_n[0,3]
   + 0.6760000228881836 simga_cut_n[0,4]
   + 0.8450000286102295 simga_cut_n[0,5]
   + 1.014000034332275 simga_cut_n[0,6]
   + 1.183000040054321 simga_cut_n[0,7]
   + 1.352000045776367 simga_cut_n[0,8]
   + 1.521000051498413 simga_cut_n[0,9] <= 1.690000057220459
 c_y_cut[1]: 0.1690000057220459 simga_cut_n[1,1]
   + 0.3380000114440918 simga_cut_n[1,2]
   + 0.5070000171661377 simga_cut_n[1,3]
   + 0.6760000228881836 simga_cut_n[1,4]
   + 0.8450000286102295 simga_cut_n[1,5]
   + 1.014000034332275 simga_cut_n[1,6]
   + 1.183000040054321 simga_cut_n[1,7]
   + 1.352000045776367 simga_cut_n[1,8]
   + 1.521000051498413 simga_cut_n[1,9] <= 1.690000057220459
 c_y_cut[2]: 0.1700000047683716 simga_cut_n[2,1]
   + 0.3400000095367431 simga_cut_n[2,2]
   + 0.5100000143051148 simga_cut_n[2,3]
   + 0.6800000190734863 simga_cut_n[2,4]
   + 0.8500000238418579 simga_cut_n[2,5]
   + 1.02000002861023 simga_cut_n[2,6] + 1.190000033378601 simga_cut_n[2,7]
   + 1.360000038146973 simga_cut_n[2,8]
   + 1.530000042915344 simga_cut_n[2,9] <= 1.700000047683716
 c_y_cut[3]: 0.1700000047683716 simga_cut_n[3,1]
   + 0.3400000095367431 simga_cut_n[3,2]
   + 0.5100000143051148 simga_cut_n[3,3]
   + 0.6800000190734863 simga_cut_n[3,4]
   + 0.8500000238418579 simga_cut_n[3,5]
   + 1.02000002861023 simga_cut_n[3,6] + 1.190000033378601 simga_cut_n[3,7]
   + 1.360000038146973 simga_cut_n[3,8]
   + 1.530000042915344 simga_cut_n[3,9] <= 1.700000047683716
 c_y_cut[4]: 0.1700000047683716 simga_cut_n[4,1]
   + 0.3400000095367431 simga_cut_n[4,2]
   + 0.5100000143051148 simga_cut_n[4,3]
   + 0.6800000190734863 simga_cut_n[4,4]
   + 0.8500000238418579 simga_cut_n[4,5]
   + 1.02000002861023 simga_cut_n[4,6] + 1.190000033378601 simga_cut_n[4,7]
   + 1.360000038146973 simga_cut_n[4,8]
   + 1.530000042915344 simga_cut_n[4,9] <= 1.700000047683716
 c_y_cut[5]: 0.2039999961853027 simga_cut_n[5,1]
   + 0.4079999923706055 simga_cut_n[5,2]
   + 0.6119999885559082 simga_cut_n[5,3]
   + 0.8159999847412109 simga_cut_n[5,4]
   + 1.019999980926514 simga_cut_n[5,5]
   + 1.223999977111816 simga_cut_n[5,6]
   + 1.427999973297119 simga_cut_n[5,7]
   + 1.631999969482422 simga_cut_n[5,8]
   + 1.835999965667725 simga_cut_n[5,9] <= 2.039999961853027
 c_y_cut[6]: 0.2039999961853027 simga_cut_n[6,1]
   + 0.4079999923706055 simga_cut_n[6,2]
   + 0.6119999885559082 simga_cut_n[6,3]
   + 0.8159999847412109 simga_cut_n[6,4]
   + 1.019999980926514 simga_cut_n[6,5]
   + 1.223999977111816 simga_cut_n[6,6]
   + 1.427999973297119 simga_cut_n[6,7]
   + 1.631999969482422 simga_cut_n[6,8]
   + 1.835999965667725 simga_cut_n[6,9] <= 2.039999961853027
 c_y_cut[7]: 0.2039999961853027 simga_cut_n[7,1]
   + 0.4079999923706055 simga_cut_n[7,2]
   + 0.6119999885559082 simga_cut_n[7,3]
   + 0.8159999847412109 simga_cut_n[7,4]
   + 1.019999980926514 simga_cut_n[7,5]
   + 1.223999977111816 simga_cut_n[7,6]
   + 1.427999973297119 simga_cut_n[7,7]
   + 1.631999969482422 simga_cut_n[7,8]
   + 1.835999965667725 simga_cut_n[7,9] <= 2.039999961853027
 c_y_cut[8]: 0.2039999961853027 simga_cut_n[8,1]
   + 0.4079999923706055 simga_cut_n[8,2]
   + 0.6119999885559082 simga_cut_n[8,3]
   + 0.8159999847412109 simga_cut_n[8,4]
   + 1.019999980926514 simga_cut_n[8,5]
   + 1.223999977111816 simga_cut_n[8,6]
   + 1.427999973297119 simga_cut_n[8,7]
   + 1.631999969482422 simga_cut_n[8,8]
   + 1.835999965667725 simga_cut_n[8,9] <= 2.039999961853027
 c_y_cut[9]: 0.2039999961853027 simga_cut_n[9,1]
   + 0.4079999923706055 simga_cut_n[9,2]
   + 0.6119999885559082 simga_cut_n[9,3]
   + 0.8159999847412109 simga_cut_n[9,4]
   + 1.019999980926514 simga_cut_n[9,5]
   + 1.223999977111816 simga_cut_n[9,6]
   + 1.427999973297119 simga_cut_n[9,7]
   + 1.631999969482422 simga_cut_n[9,8]
   + 1.835999965667725 simga_cut_n[9,9] <= 2.039999961853027
 c_y_cut[10]: 0.2380000114440918 simga_cut_n[10,1]
   + 0.4760000228881836 simga_cut_n[10,2]
   + 0.7140000343322754 simga_cut_n[10,3]
   + 0.9520000457763672 simga_cut_n[10,4]
   + 1.190000057220459 simga_cut_n[10,5]
   + 1.428000068664551 simga_cut_n[10,6]
   + 1.666000080108643 simga_cut_n[10,7]
   + 1.904000091552734 simga_cut_n[10,8]
   + 2.142000102996826 simga_cut_n[10,9] <= 2.380000114440918
 c_y_cut[11]: 0.2380000114440918 simga_cut_n[11,1]
   + 0.4760000228881836 simga_cut_n[11,2]
   + 0.7140000343322754 simga_cut_n[11,3]
   + 0.9520000457763672 simga_cut_n[11,4]
   + 1.190000057220459 simga_cut_n[11,5]
   + 1.428000068664551 simga_cut_n[11,6]
   + 1.666000080108643 simga_cut_n[11,7]
   + 1.904000091552734 simga_cut_n[11,8]
   + 2.142000102996826 simga_cut_n[11,9] <= 2.380000114440918
 c_y_cut[12]: 0.2369999885559082 simga_cut_n[12,1]
   + 0.4739999771118164 simga_cut_n[12,2]
   + 0.7109999656677246 simga_cut_n[12,3]
   + 0.9479999542236328 simga_cut_n[12,4]
   + 1.184999942779541 simga_cut_n[12,5]
   + 1.421999931335449 simga_cut_n[12,6]
   + 1.658999919891357 simga_cut_n[12,7]
   + 1.895999908447266 simga_cut_n[12,8]
   + 2.132999897003174 simga_cut_n[12,9] <= 2.369999885559082
 c_y_cut[13]: 0.2369999885559082 simga_cut_n[13,1]
   + 0.4739999771118164 simga_cut_n[13,2]
   + 0.7109999656677246 simga_cut_n[13,3]
   + 0.9479999542236328 simga_cut_n[13,4]
   + 1.184999942779541 simga_cut_n[13,5]
   + 1.421999931335449 simga_cut_n[13,6]
   + 1.658999919891357 simga_cut_n[13,7]
   + 1.895999908447266 simga_cut_n[13,8]
   + 2.132999897003174 simga_cut_n[13,9] <= 2.369999885559082
 c_y_cut[14]: 0.2369999885559082 simga_cut_n[14,1]
   + 0.4739999771118164 simga_cut_n[14,2]
   + 0.7109999656677246 simga_cut_n[14,3]
   + 0.9479999542236328 simga_cut_n[14,4]
   + 1.184999942779541 simga_cut_n[14,5]
   + 1.421999931335449 simga_cut_n[14,6]
   + 1.658999919891357 simga_cut_n[14,7]
   + 1.895999908447266 simga_cut_n[14,8]
   + 2.132999897003174 simga_cut_n[14,9] <= 2.369999885559082
 c_y_cut[15]: 0.2710000038146972 simga_cut_n[15,1]
   + 0.5420000076293945 simga_cut_n[15,2]
   + 0.8130000114440918 simga_cut_n[15,3]
   + 1.084000015258789 simga_cut_n[15,4]
   + 1.355000019073486 simga_cut_n[15,5]
   + 1.626000022888184 simga_cut_n[15,6]
   + 1.897000026702881 simga_cut_n[15,7]
   + 2.168000030517578 simga_cut_n[15,8]
   + 2.439000034332275 simga_cut_n[15,9] <= 2.710000038146973
 c_y_cut[16]: 0.2710000038146972 simga_cut_n[16,1]
   + 0.5420000076293945 simga_cut_n[16,2]
   + 0.8130000114440918 simga_cut_n[16,3]
   + 1.084000015258789 simga_cut_n[16,4]
   + 1.355000019073486 simga_cut_n[16,5]
   + 1.626000022888184 simga_cut_n[16,6]
   + 1.897000026702881 simga_cut_n[16,7]
   + 2.168000030517578 simga_cut_n[16,8]
   + 2.439000034332275 simga_cut_n[16,9] <= 2.710000038146973
 c_y_cut[17]: 0.2710000038146972 simga_cut_n[17,1]
   + 0.5420000076293945 simga_cut_n[17,2]
   + 0.8130000114440918 simga_cut_n[17,3]
   + 1.084000015258789 simga_cut_n[17,4]
   + 1.355000019073486 simga_cut_n[17,5]
   + 1.626000022888184 simga_cut_n[17,6]
   + 1.897000026702881 simga_cut_n[17,7]
   + 2.168000030517578 simga_cut_n[17,8]
   + 2.439000034332275 simga_cut_n[17,9] <= 2.710000038146973
 c_y_cut[18]: 0.2710000038146972 simga_cut_n[18,1]
   + 0.5420000076293945 simga_cut_n[18,2]
   + 0.8130000114440918 simga_cut_n[18,3]
   + 1.084000015258789 simga_cut_n[18,4]
   + 1.355000019073486 simga_cut_n[18,5]
   + 1.626000022888184 simga_cut_n[18,6]
   + 1.897000026702881 simga_cut_n[18,7]
   + 2.168000030517578 simga_cut_n[18,8]
   + 2.439000034332275 simga_cut_n[18,9] <= 2.710000038146973
 c_y_cut[19]: 0.2710000038146972 simga_cut_n[19,1]
   + 0.5420000076293945 simga_cut_n[19,2]
   + 0.8130000114440918 simga_cut_n[19,3]
   + 1.084000015258789 simga_cut_n[19,4]
   + 1.355000019073486 simga_cut_n[19,5]
   + 1.626000022888184 simga_cut_n[19,6]
   + 1.897000026702881 simga_cut_n[19,7]
   + 2.168000030517578 simga_cut_n[19,8]
   + 2.439000034332275 simga_cut_n[19,9] <= 2.710000038146973
 c_y_cut[20]: 0.3049999952316284 simga_cut_n[20,1]
   + 0.6099999904632568 simga_cut_n[20,2]
   + 0.9149999856948853 simga_cut_n[20,3]
   + 1.219999980926514 simga_cut_n[20,4]
   + 1.524999976158142 simga_cut_n[20,5]
   + 1.829999971389771 simga_cut_n[20,6]
   + 2.134999966621399 simga_cut_n[20,7]
   + 2.439999961853027 simga_cut_n[20,8]
   + 2.744999957084656 simga_cut_n[20,9] <= 3.049999952316284
 c_y_cut[21]: 0.3049999952316284 simga_cut_n[21,1]
   + 0.6099999904632568 simga_cut_n[21,2]
   + 0.9149999856948853 simga_cut_n[21,3]
   + 1.219999980926514 simga_cut_n[21,4]
   + 1.524999976158142 simga_cut_n[21,5]
   + 1.829999971389771 simga_cut_n[21,6]
   + 2.134999966621399 simga_cut_n[21,7]
   + 2.439999961853027 simga_cut_n[21,8]
   + 2.744999957084656 simga_cut_n[21,9] <= 3.049999952316284
 c_y_cut[22]: 0.3049999952316284 simga_cut_n[22,1]
   + 0.6099999904632568 simga_cut_n[22,2]
   + 0.9149999856948853 simga_cut_n[22,3]
   + 1.219999980926514 simga_cut_n[22,4]
   + 1.524999976158142 simga_cut_n[22,5]
   + 1.829999971389771 simga_cut_n[22,6]
   + 2.134999966621399 simga_cut_n[22,7]
   + 2.439999961853027 simga_cut_n[22,8]
   + 2.744999957084656 simga_cut_n[22,9] <= 3.049999952316284
 c_y_cut[23]: 0.3049999952316284 simga_cut_n[23,1]
   + 0.6099999904632568 simga_cut_n[23,2]
   + 0.9149999856948853 simga_cut_n[23,3]
   + 1.219999980926514 simga_cut_n[23,4]
   + 1.524999976158142 simga_cut_n[23,5]
   + 1.829999971389771 simga_cut_n[23,6]
   + 2.134999966621399 simga_cut_n[23,7]
   + 2.439999961853027 simga_cut_n[23,8]
   + 2.744999957084656 simga_cut_n[23,9] <= 3.049999952316284
 c_y_cut[24]: 0.3049999952316284 simga_cut_n[24,1]
   + 0.6099999904632568 simga_cut_n[24,2]
   + 0.9149999856948853 simga_cut_n[24,3]
   + 1.219999980926514 simga_cut_n[24,4]
   + 1.524999976158142 simga_cut_n[24,5]
   + 1.829999971389771 simga_cut_n[24,6]
   + 2.134999966621399 simga_cut_n[24,7]
   + 2.439999961853027 simga_cut_n[24,8]
   + 2.744999957084656 simga_cut_n[24,9] <= 3.049999952316284
 c_y_cut[25]: 0.3390000104904175 simga_cut_n[25,1]
   + 0.678000020980835 simga_cut_n[25,2]
   + 1.017000031471252 simga_cut_n[25,3]
   + 1.35600004196167 simga_cut_n[25,4]
   + 1.695000052452087 simga_cut_n[25,5]
   + 2.034000062942505 simga_cut_n[25,6]
   + 2.373000073432922 simga_cut_n[25,7]
   + 2.71200008392334 simga_cut_n[25,8]
   + 3.051000094413757 simga_cut_n[25,9] <= 3.390000104904175
 c_y_cut[26]: 0.3049999952316284 simga_cut_n[26,1]
   + 0.6099999904632568 simga_cut_n[26,2]
   + 0.9149999856948853 simga_cut_n[26,3]
   + 1.219999980926514 simga_cut_n[26,4]
   + 1.524999976158142 simga_cut_n[26,5]
   + 1.829999971389771 simga_cut_n[26,6]
   + 2.134999966621399 simga_cut_n[26,7]
   + 2.439999961853027 simga_cut_n[26,8]
   + 2.744999957084656 simga_cut_n[26,9] <= 3.049999952316284
 c_y_cut[27]: 1.119999980926514 simga_cut_n[27,1]
   + 2.239999961853028 simga_cut_n[27,2]
   + 3.359999942779541 simga_cut_n[27,3]
   + 4.479999923706055 simga_cut_n[27,4]
   + 5.599999904632568 simga_cut_n[27,5]
   + 6.719999885559082 simga_cut_n[27,6]
   + 7.839999866485596 simga_cut_n[27,7]
   + 8.95999984741211 simga_cut_n[27,8]
   + 10.07999982833862 simga_cut_n[27,9] <= 11.19999980926514
 c_y_cut[28]: 2.025 simga_cut_n[28,1] + 4.05 simga_cut_n[28,2]
   + 6.075 simga_cut_n[28,3] + 8.1 simga_cut_n[28,4]
   + 10.125 simga_cut_n[28,5] + 12.15 simga_cut_n[28,6]
   + 14.175 simga_cut_n[28,7] + 16.2 simga_cut_n[28,8]
   + 18.225 simga_cut_n[28,9] <= 20.25
 c_y_cut[29]: 3.22599983215332 simga_cut_n[29,1]
   + 6.451999664306641 simga_cut_n[29,2]
   + 9.677999496459961 simga_cut_n[29,3]
   + 12.90399932861328 simga_cut_n[29,4]
   + 16.1299991607666 simga_cut_n[29,5]
   + 19.35599899291992 simga_cut_n[29,6]
   + 22.58199882507324 simga_cut_n[29,7]
   + 25.80799865722656 simga_cut_n[29,8]
   + 29.03399848937988 simga_cut_n[29,9] <= 32.2599983215332
 c_y_cut[30]: 2.403000068664551 simga_cut_n[30,1]
   + 4.806000137329102 simga_cut_n[30,2]
   + 7.209000205993652 simga_cut_n[30,3]
   + 9.612000274658204 simga_cut_n[30,4]
   + 12.01500034332275 simga_cut_n[30,5]
   + 14.4180004119873 simga_cut_n[30,6]
   + 16.82100048065185 simga_cut_n[30,7]
   + 19.22400054931641 simga_cut_n[30,8]
   + 21.62700061798096 simga_cut_n[30,9] <= 24.03000068664551
 c_y_cut[31]: 6.109000015258789 simga_cut_n[31,1]
   + 12.21800003051758 simga_cut_n[31,2]
   + 18.32700004577637 simga_cut_n[31,3]
   + 24.43600006103516 simga_cut_n[31,4]
   + 30.54500007629395 simga_cut_n[31,5]
   + 36.65400009155273 simga_cut_n[31,6]
   + 42.76300010681152 simga_cut_n[31,7]
   + 48.87200012207031 simga_cut_n[31,8]
   + 54.9810001373291 simga_cut_n[31,9] <= 61.09000015258789
 c_y_cut[32]: 10.04300003051758 simga_cut_n[32,1]
   + 20.08600006103516 simga_cut_n[32,2]
   + 30.12900009155273 simga_cut_n[32,3]
   + 40.17200012207032 simga_cut_n[32,4]
   + 50.21500015258789 simga_cut_n[32,5]
   + 60.25800018310547 simga_cut_n[32,6]
   + 70.30100021362304 simga_cut_n[32,7]
   + 80.34400024414063 simga_cut_n[32,8]
   + 90.38700027465821 simga_cut_n[32,9] <= 100.4300003051758
 c_y_cut[33]: 13.48500061035156 simga_cut_n[33,1]
   + 26.97000122070313 simga_cut_n[33,2]
   + 40.45500183105469 simga_cut_n[33,3]
   + 53.94000244140625 simga_cut_n[33,4]
   + 67.42500305175781 simga_cut_n[33,5]
   + 80.91000366210938 simga_cut_n[33,6]
   + 94.39500427246094 simga_cut_n[33,7]
   + 107.8800048828125 simga_cut_n[33,8]
   + 121.3650054931641 simga_cut_n[33,9] <= 134.8500061035156
 c_y_cut[34]: 15.47200012207031 simga_cut_n[34,1]
   + 30.94400024414063 simga_cut_n[34,2]
   + 46.41600036621094 simga_cut_n[34,3]
   + 61.88800048828125 simga_cut_n[34,4]
   + 77.36000061035156 simga_cut_n[34,5]
   + 92.83200073242188 simga_cut_n[34,6]
   + 108.3040008544922 simga_cut_n[34,7]
   + 123.7760009765625 simga_cut_n[34,8]
   + 139.2480010986328 simga_cut_n[34,9] <= 154.7200012207031
 c_y_cut[35]: 16.30223693847656 simga_cut_n[35,1]
   + 32.60447387695312 simga_cut_n[35,2]
   + 48.90671081542969 simga_cut_n[35,3]
   + 65.20894775390624 simga_cut_n[35,4]
   + 81.51118469238281 simga_cut_n[35,5]
   + 97.81342163085938 simga_cut_n[35,6]
   + 114.1156585693359 simga_cut_n[35,7]
   + 130.4178955078125 simga_cut_n[35,8]
   + 146.7201324462891 simga_cut_n[35,9] <= 163.0223693847656
 c_y_cut[36]: 17.67989807128906 simga_cut_n[36,1]
   + 35.35979614257813 simga_cut_n[36,2]
   + 53.03969421386719 simga_cut_n[36,3]
   + 70.71959228515625 simga_cut_n[36,4]
   + 88.39949035644531 simga_cut_n[36,5]
   + 106.0793884277344 simga_cut_n[36,6]
   + 123.7592864990234 simga_cut_n[36,7]
   + 141.4391845703125 simga_cut_n[36,8]
   + 159.1190826416016 simga_cut_n[36,9] <= 176.7989807128906
 c_y_cut[37]: 18.15235443115234 simga_cut_n[37,1]
   + 36.30470886230469 simga_cut_n[37,2]
   + 54.45706329345703 simga_cut_n[37,3]
   + 72.60941772460937 simga_cut_n[37,4]
   + 90.76177215576172 simga_cut_n[37,5]
   + 108.9141265869141 simga_cut_n[37,6]
   + 127.0664810180664 simga_cut_n[37,7]
   + 145.2188354492187 simga_cut_n[37,8]
   + 163.3711898803711 simga_cut_n[37,9] <= 181.5235443115234
 c_y_cut[38]: 18.81719269193793 simga_cut_n[38,1]
   + 37.63438538387587 simga_cut_n[38,2]
   + 56.4515780758138 simga_cut_n[38,3]
   + 75.26877076775173 simga_cut_n[38,4]
   + 94.08596345968967 simga_cut_n[38,5]
   + 112.9031561516276 simga_cut_n[38,6]
   + 131.7203488435655 simga_cut_n[38,7]
   + 150.5375415355035 simga_cut_n[38,8]
   + 169.3547342274414 simga_cut_n[38,9] <= 188.1719269193793
 c_y_cut[39]: 17.90970977035714 simga_cut_n[39,1]
   + 35.81941954071429 simga_cut_n[39,2]
   + 53.72912931107142 simga_cut_n[39,3]
   + 71.63883908142857 simga_cut_n[39,4]
   + 89.54854885178571 simga_cut_n[39,5]
   + 107.4582586221428 simga_cut_n[39,6] + 125.3679683925 simga_cut_n[39,7]
   + 143.2776781628571 simga_cut_n[39,8]
   + 161.1873879332143 simga_cut_n[39,9] <= 179.0970977035714
 c_y_cut[40]: 21.68679973842232 simga_cut_n[40,1]
   + 43.37359947684463 simga_cut_n[40,2]
   + 65.06039921526694 simga_cut_n[40,3]
   + 86.74719895368926 simga_cut_n[40,4]
   + 108.4339986921116 simga_cut_n[40,5]
   + 130.1207984305339 simga_cut_n[40,6]
   + 151.8075981689562 simga_cut_n[40,7]
   + 173.4943979073785 simga_cut_n[40,8]
   + 195.1811976458008 simga_cut_n[40,9] <= 216.8679973842231
 c_y_cut[41]: 23.03771037501046 simga_cut_n[41,1]
   + 46.07542075002091 simga_cut_n[41,2]
   + 69.11313112503136 simga_cut_n[41,3]
   + 92.15084150004182 simga_cut_n[41,4]
   + 115.1885518750523 simga_cut_n[41,5]
   + 138.2262622500627 simga_cut_n[41,6]
   + 161.2639726250732 simga_cut_n[41,7]
   + 184.3016830000836 simga_cut_n[41,8]
   + 207.3393933750941 simga_cut_n[41,9] <= 230.3771037501045
 c_y_cut[42]: 16.47894415424698 simga_cut_n[42,1]
   + 32.95788830849396 simga_cut_n[42,2]
   + 49.43683246274095 simga_cut_n[42,3]
   + 65.91577661698793 simga_cut_n[42,4]
   + 82.39472077123492 simga_cut_n[42,5]
   + 98.8736649254819 simga_cut_n[42,6]
   + 115.3526090797289 simga_cut_n[42,7]
   + 131.8315532339759 simga_cut_n[42,8]
   + 148.3104973882228 simga_cut_n[42,9] <= 164.7894415424698
 c_y_cut[43]: 17.4226204773704 simga_cut_n[43,1]
   + 34.8452409547408 simga_cut_n[43,2]
   + 52.2678614321112 simga_cut_n[43,3]
   + 69.6904819094816 simga_cut_n[43,4]
   + 87.11310238685201 simga_cut_n[43,5]
   + 104.5357228642224 simga_cut_n[43,6]
   + 121.9583433415928 simga_cut_n[43,7]
   + 139.3809638189632 simga_cut_n[43,8]
   + 156.8035842963336 simga_cut_n[43,9] <= 174.226204773704
 c_y_cut[44]: 23.75216920991493 simga_cut_n[44,1]
   + 47.50433841982986 simga_cut_n[44,2]
   + 71.25650762974479 simga_cut_n[44,3]
   + 95.00867683965971 simga_cut_n[44,4]
   + 118.7608460495746 simga_cut_n[44,5]
   + 142.5130152594896 simga_cut_n[44,6]
   + 166.2651844694045 simga_cut_n[44,7]
   + 190.0173536793194 simga_cut_n[44,8]
   + 213.7695228892343 simga_cut_n[44,9] <= 237.5216920991493
 c_y_cut[45]: 24.49579679190596 simga_cut_n[45,1]
   + 48.99159358381192 simga_cut_n[45,2]
   + 73.48739037571788 simga_cut_n[45,3]
   + 97.98318716762384 simga_cut_n[45,4]
   + 122.4789839595298 simga_cut_n[45,5]
   + 146.9747807514358 simga_cut_n[45,6]
   + 171.4705775433417 simga_cut_n[45,7]
   + 195.9663743352477 simga_cut_n[45,8]
   + 220.4621711271536 simga_cut_n[45,9] <= 244.9579679190596
 c_y_cut[46]: 28.38617568876003 simga_cut_n[46,1]
   + 56.77235137752007 simga_cut_n[46,2]
   + 85.1585270662801 simga_cut_n[46,3]
   + 113.5447027550401 simga_cut_n[46,4]
   + 141.9308784438002 simga_cut_n[46,5]
   + 170.3170541325602 simga_cut_n[46,6]
   + 198.7032298213202 simga_cut_n[46,7]
   + 227.0894055100803 simga_cut_n[46,8]
   + 255.4755811988403 simga_cut_n[46,9] <= 283.8617568876003
 c_y_cut[47]: 28.8166116363088 simga_cut_n[47,1]
   + 57.6332232726176 simga_cut_n[47,2]
   + 86.44983490892641 simga_cut_n[47,3]
   + 115.2664465452352 simga_cut_n[47,4]
   + 144.083058181544 simga_cut_n[47,5]
   + 172.8996698178528 simga_cut_n[47,6]
   + 201.7162814541616 simga_cut_n[47,7]
   + 230.5328930904704 simga_cut_n[47,8]
   + 259.3495047267792 simga_cut_n[47,9] <= 288.166116363088
 c_y_cut[48]: 30.3288536067829 simga_cut_n[48,1]
   + 60.6577072135658 simga_cut_n[48,2]
   + 90.9865608203487 simga_cut_n[48,3]
   + 121.3154144271316 simga_cut_n[48,4]
   + 151.6442680339145 simga_cut_n[48,5]
   + 181.9731216406974 simga_cut_n[48,6]
   + 212.3019752474803 simga_cut_n[48,7]
   + 242.6308288542632 simga_cut_n[48,8]
   + 272.9596824610461 simga_cut_n[48,9] <= 303.288536067829
 c_y_cut[49]: 32.12749037692363 simga_cut_n[49,1]
   + 64.25498075384726 simga_cut_n[49,2]
   + 96.38247113077091 simga_cut_n[49,3]
   + 128.5099615076945 simga_cut_n[49,4]
   + 160.6374518846182 simga_cut_n[49,5]
   + 192.7649422615418 simga_cut_n[49,6]
   + 224.8924326384654 simga_cut_n[49,7]
   + 257.0199230153891 simga_cut_n[49,8]
   + 289.1474133923127 simga_cut_n[49,9] <= 321.2749037692363
 c_y_cut[50]: 30.46193185373683 simga_cut_n[50,1]
   + 60.92386370747365 simga_cut_n[50,2]
   + 91.38579556121047 simga_cut_n[50,3]
   + 121.8477274149473 simga_cut_n[50,4]
   + 152.3096592686841 simga_cut_n[50,5]
   + 182.7715911224209 simga_cut_n[50,6]
   + 213.2335229761578 simga_cut_n[50,7]
   + 243.6954548298946 simga_cut_n[50,8]
   + 274.1573866836314 simga_cut_n[50,9] <= 304.6193185373683
 c_y_cut[51]: 30.91058977901737 simga_cut_n[51,1]
   + 61.82117955803474 simga_cut_n[51,2]
   + 92.7317693370521 simga_cut_n[51,3]
   + 123.6423591160695 simga_cut_n[51,4]
   + 154.5529488950868 simga_cut_n[51,5]
   + 185.4635386741042 simga_cut_n[51,6]
   + 216.3741284531216 simga_cut_n[51,7]
   + 247.2847182321389 simga_cut_n[51,8]
   + 278.1953080111563 simga_cut_n[51,9] <= 309.1058977901737
 c_y_cut[52]: 31.14318937414892 simga_cut_n[52,1]
   + 62.28637874829784 simga_cut_n[52,2]
   + 93.42956812244677 simga_cut_n[52,3]
   + 124.5727574965957 simga_cut_n[52,4]
   + 155.7159468707446 simga_cut_n[52,5]
   + 186.8591362448935 simga_cut_n[52,6]
   + 218.0023256190424 simga_cut_n[52,7]
   + 249.1455149931914 simga_cut_n[52,8]
   + 280.2887043673403 simga_cut_n[52,9] <= 311.4318937414892
 c_y_cut[53]: 30.84708162501029 simga_cut_n[53,1]
   + 61.69416325002057 simga_cut_n[53,2]
   + 92.54124487503086 simga_cut_n[53,3]
   + 123.3883265000411 simga_cut_n[53,4]
   + 154.2354081250514 simga_cut_n[53,5]
   + 185.0824897500617 simga_cut_n[53,6]
   + 215.929571375072 simga_cut_n[53,7]
   + 246.7766530000823 simga_cut_n[53,8]
   + 277.6237346250926 simga_cut_n[53,9] <= 308.4708162501029
 c_y_cut[54]: 31.43929834392985 simga_cut_n[54,1]
   + 62.87859668785969 simga_cut_n[54,2]
   + 94.31789503178955 simga_cut_n[54,3]
   + 125.7571933757194 simga_cut_n[54,4]
   + 157.1964917196492 simga_cut_n[54,5]
   + 188.6357900635791 simga_cut_n[54,6]
   + 220.0750884075089 simga_cut_n[54,7]
   + 251.5143867514388 simga_cut_n[54,8]
   + 282.9536850953686 simga_cut_n[54,9] <= 314.3929834392985
 c_y_cut[55]: 29.82442895942964 simga_cut_n[55,1]
   + 59.64885791885928 simga_cut_n[55,2]
   + 89.47328687828892 simga_cut_n[55,3]
   + 119.2977158377186 simga_cut_n[55,4]
   + 149.1221447971482 simga_cut_n[55,5]
   + 178.9465737565778 simga_cut_n[55,6]
   + 208.7710027160075 simga_cut_n[55,7]
   + 238.5954316754371 simga_cut_n[55,8]
   + 268.4198606348667 simga_cut_n[55,9] <= 298.2442895942964
 c_y_cut[56]: 25.28332839737743 simga_cut_n[56,1]
   + 50.56665679475486 simga_cut_n[56,2]
   + 75.84998519213229 simga_cut_n[56,3]
   + 101.1333135895097 simga_cut_n[56,4]
   + 126.4166419868872 simga_cut_n[56,5]
   + 151.6999703842646 simga_cut_n[56,6]
   + 176.983298781642 simga_cut_n[56,7]
   + 202.2666271790195 simga_cut_n[56,8]
   + 227.5499555763969 simga_cut_n[56,9] <= 252.8332839737743
 c_y_cut[57]: 25.98666045995219 simga_cut_n[57,1]
   + 51.97332091990438 simga_cut_n[57,2]
   + 77.95998137985657 simga_cut_n[57,3]
   + 103.9466418398088 simga_cut_n[57,4]
   + 129.933302299761 simga_cut_n[57,5]
   + 155.9199627597131 simga_cut_n[57,6]
   + 181.9066232196653 simga_cut_n[57,7]
   + 207.8932836796175 simga_cut_n[57,8]
   + 233.8799441395697 simga_cut_n[57,9] <= 259.8666045995219
 c_y_cut[58]: 25.54966036925494 simga_cut_n[58,1]
   + 51.09932073850989 simga_cut_n[58,2]
   + 76.64898110776483 simga_cut_n[58,3]
   + 102.1986414770198 simga_cut_n[58,4]
   + 127.7483018462747 simga_cut_n[58,5]
   + 153.2979622155297 simga_cut_n[58,6]
   + 178.8476225847846 simga_cut_n[58,7]
   + 204.3972829540396 simga_cut_n[58,8]
   + 229.9469433232945 simga_cut_n[58,9] <= 255.4966036925494
 c_y_cut[59]: 27.44065962706981 simga_cut_n[59,1]
   + 54.88131925413961 simga_cut_n[59,2]
   + 82.32197888120942 simga_cut_n[59,3]
   + 109.7626385082792 simga_cut_n[59,4]
   + 137.203298135349 simga_cut_n[59,5]
   + 164.6439577624188 simga_cut_n[59,6]
   + 192.0846173894886 simga_cut_n[59,7]
   + 219.5252770165584 simga_cut_n[59,8]
   + 246.9659366436283 simga_cut_n[59,9] <= 274.4065962706981
 c_y_cut[60]: 24.92622129187907 simga_cut_n[60,1]
   + 49.85244258375813 simga_cut_n[60,2]
   + 74.77866387563719 simga_cut_n[60,3]
   + 99.70488516751627 simga_cut_n[60,4]
   + 124.6311064593953 simga_cut_n[60,5]
   + 149.5573277512744 simga_cut_n[60,6]
   + 174.4835490431535 simga_cut_n[60,7]
   + 199.4097703350325 simga_cut_n[60,8]
   + 224.3359916269116 simga_cut_n[60,9] <= 249.2622129187907
 c_y_cut[61]: 23.20547884714184 simga_cut_n[61,1]
   + 46.41095769428368 simga_cut_n[61,2]
   + 69.61643654142551 simga_cut_n[61,3]
   + 92.82191538856736 simga_cut_n[61,4]
   + 116.0273942357092 simga_cut_n[61,5]
   + 139.232873082851 simga_cut_n[61,6]
   + 162.4383519299929 simga_cut_n[61,7]
   + 185.6438307771347 simga_cut_n[61,8]
   + 208.8493096242766 simga_cut_n[61,9] <= 232.0547884714184
 c_y_cut[62]: 18.55271517812684 simga_cut_n[62,1]
   + 37.10543035625368 simga_cut_n[62,2]
   + 55.65814553438052 simga_cut_n[62,3]
   + 74.21086071250735 simga_cut_n[62,4]
   + 92.7635758906342 simga_cut_n[62,5]
   + 111.316291068761 simga_cut_n[62,6]
   + 129.8690062468879 simga_cut_n[62,7]
   + 148.4217214250147 simga_cut_n[62,8]
   + 166.9744366031416 simga_cut_n[62,9] <= 185.5271517812684
 c_y_cut[63]: 25.11913585499702 simga_cut_n[63,1]
   + 50.23827170999405 simga_cut_n[63,2]
   + 75.35740756499106 simga_cut_n[63,3]
   + 100.4765434199881 simga_cut_n[63,4]
   + 125.5956792749851 simga_cut_n[63,5]
   + 150.7148151299821 simga_cut_n[63,6]
   + 175.8339509849792 simga_cut_n[63,7]
   + 200.9530868399762 simga_cut_n[63,8]
   + 226.0722226949732 simga_cut_n[63,9] <= 251.1913585499702
 c_y_cut[64]: 19.16049325766229 simga_cut_n[64,1]
   + 38.32098651532458 simga_cut_n[64,2]
   + 57.48147977298688 simga_cut_n[64,3]
   + 76.64197303064917 simga_cut_n[64,4]
   + 95.80246628831146 simga_cut_n[64,5]
   + 114.9629595459738 simga_cut_n[64,6]
   + 134.123452803636 simga_cut_n[64,7]
   + 153.2839460612983 simga_cut_n[64,8]
   + 172.4444393189606 simga_cut_n[64,9] <= 191.6049325766229
 c_y_cut[65]: 16.15287235167553 simga_cut_n[65,1]
   + 32.30574470335107 simga_cut_n[65,2]
   + 48.45861705502661 simga_cut_n[65,3]
   + 64.61148940670213 simga_cut_n[65,4]
   + 80.76436175837767 simga_cut_n[65,5]
   + 96.91723411005322 simga_cut_n[65,6]
   + 113.0701064617288 simga_cut_n[65,7]
   + 129.2229788134043 simga_cut_n[65,8]
   + 145.3758511650798 simga_cut_n[65,9] <= 161.5287235167553
 c_y_cut[66]: 13.18351872664955 simga_cut_n[66,1]
   + 26.3670374532991 simga_cut_n[66,2]
   + 39.55055617994865 simga_cut_n[66,3]
   + 52.7340749065982 simga_cut_n[66,4]
   + 65.91759363324775 simga_cut_n[66,5]
   + 79.1011123598973 simga_cut_n[66,6]
   + 92.28463108654685 simga_cut_n[66,7]
   + 105.4681498131964 simga_cut_n[66,8]
   + 118.6516685398459 simga_cut_n[66,9] <= 131.8351872664955
 c_y_cut[67]: 23.5277914127786 simga_cut_n[67,1]
   + 47.05558282555721 simga_cut_n[67,2]
   + 70.58337423833581 simga_cut_n[67,3]
   + 94.11116565111442 simga_cut_n[67,4]
   + 117.638957063893 simga_cut_n[67,5]
   + 141.1667484766716 simga_cut_n[67,6]
   + 164.6945398894502 simga_cut_n[67,7]
   + 188.2223313022288 simga_cut_n[67,8]
   + 211.7501227150074 simga_cut_n[67,9] <= 235.277914127786
 c_y_cut[68]: 10.69800033569336 simga_cut_n[68,1]
   + 21.39600067138672 simga_cut_n[68,2]
   + 32.09400100708008 simga_cut_n[68,3]
   + 42.79200134277344 simga_cut_n[68,4]
   + 53.4900016784668 simga_cut_n[68,5]
   + 64.18800201416016 simga_cut_n[68,6]
   + 74.88600234985351 simga_cut_n[68,7]
   + 85.58400268554688 simga_cut_n[68,8]
   + 96.28200302124023 simga_cut_n[68,9] <= 106.9800033569336
 c_y_cut[69]: 11.94499969482422 simga_cut_n[69,1]
   + 23.88999938964844 simga_cut_n[69,2]
   + 35.83499908447266 simga_cut_n[69,3]
   + 47.77999877929688 simga_cut_n[69,4]
   + 59.72499847412109 simga_cut_n[69,5]
   + 71.66999816894531 simga_cut_n[69,6]
   + 83.61499786376953 simga_cut_n[69,7]
   + 95.55999755859375 simga_cut_n[69,8]
   + 107.504997253418 simga_cut_n[69,9] <= 119.4499969482422
 c_y_cut[70]: 5.481000137329102 simga_cut_n[70,1]
   + 10.9620002746582 simga_cut_n[70,2]
   + 16.44300041198731 simga_cut_n[70,3]
   + 21.92400054931641 simga_cut_n[70,4]
   + 27.40500068664551 simga_cut_n[70,5]
   + 32.88600082397461 simga_cut_n[70,6]
   + 38.36700096130371 simga_cut_n[70,7]
   + 43.84800109863281 simga_cut_n[70,8]
   + 49.32900123596191 simga_cut_n[70,9] <= 54.81000137329102
 c_y_cut[71]: 6.086000061035156 simga_cut_n[71,1]
   + 12.17200012207031 simga_cut_n[71,2]
   + 18.25800018310547 simga_cut_n[71,3]
   + 24.34400024414062 simga_cut_n[71,4]
   + 30.43000030517578 simga_cut_n[71,5]
   + 36.51600036621094 simga_cut_n[71,6]
   + 42.6020004272461 simga_cut_n[71,7]
   + 48.68800048828125 simga_cut_n[71,8]
   + 54.7740005493164 simga_cut_n[71,9] <= 60.86000061035156
 c_y_cut[72]: 6.444999694824219 simga_cut_n[72,1]
   + 12.88999938964844 simga_cut_n[72,2]
   + 19.33499908447266 simga_cut_n[72,3]
   + 25.77999877929688 simga_cut_n[72,4]
   + 32.22499847412109 simga_cut_n[72,5]
   + 38.66999816894531 simga_cut_n[72,6]
   + 45.11499786376953 simga_cut_n[72,7]
   + 51.55999755859375 simga_cut_n[72,8]
   + 58.00499725341797 simga_cut_n[72,9] <= 64.44999694824219
 c_y_cut[73]: 6.113999938964843 simga_cut_n[73,1]
   + 12.22799987792969 simga_cut_n[73,2]
   + 18.34199981689453 simga_cut_n[73,3]
   + 24.45599975585937 simga_cut_n[73,4]
   + 30.56999969482422 simga_cut_n[73,5]
   + 36.68399963378906 simga_cut_n[73,6]
   + 42.79799957275391 simga_cut_n[73,7]
   + 48.91199951171875 simga_cut_n[73,8]
   + 55.02599945068359 simga_cut_n[73,9] <= 61.13999938964844
 c_y_cut[74]: 3.130999946594238 simga_cut_n[74,1]
   + 6.261999893188476 simga_cut_n[74,2]
   + 9.392999839782714 simga_cut_n[74,3]
   + 12.52399978637695 simga_cut_n[74,4]
   + 15.65499973297119 simga_cut_n[74,5]
   + 18.78599967956543 simga_cut_n[74,6]
   + 21.91699962615967 simga_cut_n[74,7]
   + 25.04799957275391 simga_cut_n[74,8]
   + 28.17899951934815 simga_cut_n[74,9] <= 31.30999946594238
 c_y_cut[75]: 1.214999961853027 simga_cut_n[75,1]
   + 2.429999923706055 simga_cut_n[75,2]
   + 3.644999885559082 simga_cut_n[75,3]
   + 4.85999984741211 simga_cut_n[75,4]
   + 6.074999809265137 simga_cut_n[75,5]
   + 7.289999771118164 simga_cut_n[75,6]
   + 8.504999732971191 simga_cut_n[75,7]
   + 9.719999694824219 simga_cut_n[75,8]
   + 10.93499965667725 simga_cut_n[75,9] <= 12.14999961853027
 c_y_cut[76]: 0.4710000038146973 simga_cut_n[76,1]
   + 0.9420000076293945 simga_cut_n[76,2]
   + 1.413000011444092 simga_cut_n[76,3]
   + 1.884000015258789 simga_cut_n[76,4]
   + 2.355000019073486 simga_cut_n[76,5]
   + 2.826000022888183 simga_cut_n[76,6]
   + 3.297000026702881 simga_cut_n[76,7]
   + 3.768000030517578 simga_cut_n[76,8]
   + 4.239000034332276 simga_cut_n[76,9] <= 4.710000038146973
 c_y_cut[77]: 0.0360000014305115 simga_cut_n[77,1]
   + 0.072000002861023 simga_cut_n[77,2]
   + 0.1080000042915344 simga_cut_n[77,3]
   + 0.1440000057220459 simga_cut_n[77,4]
   + 0.1800000071525574 simga_cut_n[77,5]
   + 0.2160000085830688 simga_cut_n[77,6]
   + 0.2520000100135803 simga_cut_n[77,7]
   + 0.2880000114440918 simga_cut_n[77,8]
   + 0.3240000128746033 simga_cut_n[77,9] <= 0.3600000143051147
 c_y_cut[78]: 0.0360000014305115 simga_cut_n[78,1]
   + 0.072000002861023 simga_cut_n[78,2]
   + 0.1080000042915344 simga_cut_n[78,3]
   + 0.1440000057220459 simga_cut_n[78,4]
   + 0.1800000071525574 simga_cut_n[78,5]
   + 0.2160000085830688 simga_cut_n[78,6]
   + 0.2520000100135803 simga_cut_n[78,7]
   + 0.2880000114440918 simga_cut_n[78,8]
   + 0.3240000128746033 simga_cut_n[78,9] <= 0.3600000143051147
 c_y_cut[79]: 0.0360000014305115 simga_cut_n[79,1]
   + 0.072000002861023 simga_cut_n[79,2]
   + 0.1080000042915344 simga_cut_n[79,3]
   + 0.1440000057220459 simga_cut_n[79,4]
   + 0.1800000071525574 simga_cut_n[79,5]
   + 0.2160000085830688 simga_cut_n[79,6]
   + 0.2520000100135803 simga_cut_n[79,7]
   + 0.2880000114440918 simga_cut_n[79,8]
   + 0.3240000128746033 simga_cut_n[79,9] <= 0.3600000143051147
 c_y_cut[80]: 0.072000002861023 simga_cut_n[80,1]
   + 0.1440000057220459 simga_cut_n[80,2]
   + 0.2160000085830688 simga_cut_n[80,3]
   + 0.2880000114440918 simga_cut_n[80,4]
   + 0.3600000143051147 simga_cut_n[80,5]
   + 0.4320000171661377 simga_cut_n[80,6]
   + 0.5040000200271606 simga_cut_n[80,7]
   + 0.5760000228881836 simga_cut_n[80,8]
   + 0.6480000257492066 simga_cut_n[80,9] <= 0.7200000286102295
 c_y_cut[81]: 0.072000002861023 simga_cut_n[81,1]
   + 0.1440000057220459 simga_cut_n[81,2]
   + 0.2160000085830688 simga_cut_n[81,3]
   + 0.2880000114440918 simga_cut_n[81,4]
   + 0.3600000143051147 simga_cut_n[81,5]
   + 0.4320000171661377 simga_cut_n[81,6]
   + 0.5040000200271606 simga_cut_n[81,7]
   + 0.5760000228881836 simga_cut_n[81,8]
   + 0.6480000257492066 simga_cut_n[81,9] <= 0.7200000286102295
 c_y_cut[82]: 0.0629999995231628 simga_cut_n[82,1]
   + 0.1259999990463257 simga_cut_n[82,2]
   + 0.1889999985694885 simga_cut_n[82,3]
   + 0.2519999980926514 simga_cut_n[82,4]
   + 0.3149999976158142 simga_cut_n[82,5]
   + 0.377999997138977 simga_cut_n[82,6]
   + 0.4409999966621399 simga_cut_n[82,7]
   + 0.5039999961853028 simga_cut_n[82,8]
   + 0.5669999957084656 simga_cut_n[82,9] <= 0.6299999952316284
 c_y_cut[83]: 0.0629999995231628 simga_cut_n[83,1]
   + 0.1259999990463257 simga_cut_n[83,2]
   + 0.1889999985694885 simga_cut_n[83,3]
   + 0.2519999980926514 simga_cut_n[83,4]
   + 0.3149999976158142 simga_cut_n[83,5]
   + 0.377999997138977 simga_cut_n[83,6]
   + 0.4409999966621399 simga_cut_n[83,7]
   + 0.5039999961853028 simga_cut_n[83,8]
   + 0.5669999957084656 simga_cut_n[83,9] <= 0.6299999952316284
 c_y_cut[84]: 0.1070000052452087 simga_cut_n[84,1]
   + 0.2140000104904175 simga_cut_n[84,2]
   + 0.3210000157356262 simga_cut_n[84,3]
   + 0.4280000209808349 simga_cut_n[84,4]
   + 0.5350000262260437 simga_cut_n[84,5]
   + 0.6420000314712524 simga_cut_n[84,6]
   + 0.7490000367164612 simga_cut_n[84,7]
   + 0.8560000419616699 simga_cut_n[84,8]
   + 0.9630000472068787 simga_cut_n[84,9] <= 1.070000052452087
 c_y_cut[85]: 0.1070000052452087 simga_cut_n[85,1]
   + 0.2140000104904175 simga_cut_n[85,2]
   + 0.3210000157356262 simga_cut_n[85,3]
   + 0.4280000209808349 simga_cut_n[85,4]
   + 0.5350000262260437 simga_cut_n[85,5]
   + 0.6420000314712524 simga_cut_n[85,6]
   + 0.7490000367164612 simga_cut_n[85,7]
   + 0.8560000419616699 simga_cut_n[85,8]
   + 0.9630000472068787 simga_cut_n[85,9] <= 1.070000052452087
 c_y_cut[86]: 0.1070000052452087 simga_cut_n[86,1]
   + 0.2140000104904175 simga_cut_n[86,2]
   + 0.3210000157356262 simga_cut_n[86,3]
   + 0.4280000209808349 simga_cut_n[86,4]
   + 0.5350000262260437 simga_cut_n[86,5]
   + 0.6420000314712524 simga_cut_n[86,6]
   + 0.7490000367164612 simga_cut_n[86,7]
   + 0.8560000419616699 simga_cut_n[86,8]
   + 0.9630000472068787 simga_cut_n[86,9] <= 1.070000052452087
 c_y_cut[87]: 0.1070000052452087 simga_cut_n[87,1]
   + 0.2140000104904175 simga_cut_n[87,2]
   + 0.3210000157356262 simga_cut_n[87,3]
   + 0.4280000209808349 simga_cut_n[87,4]
   + 0.5350000262260437 simga_cut_n[87,5]
   + 0.6420000314712524 simga_cut_n[87,6]
   + 0.7490000367164612 simga_cut_n[87,7]
   + 0.8560000419616699 simga_cut_n[87,8]
   + 0.9630000472068787 simga_cut_n[87,9] <= 1.070000052452087
 c_y_cut[88]: 0.1399999976158142 simga_cut_n[88,1]
   + 0.2799999952316284 simga_cut_n[88,2]
   + 0.4199999928474426 simga_cut_n[88,3]
   + 0.5599999904632569 simga_cut_n[88,4]
   + 0.699999988079071 simga_cut_n[88,5]
   + 0.8399999856948852 simga_cut_n[88,6]
   + 0.9799999833106995 simga_cut_n[88,7]
   + 1.119999980926514 simga_cut_n[88,8]
   + 1.259999978542328 simga_cut_n[88,9] <= 1.399999976158142
 c_y_cut[89]: 0.1320000052452087 simga_cut_n[89,1]
   + 0.2640000104904175 simga_cut_n[89,2]
   + 0.3960000157356262 simga_cut_n[89,3]
   + 0.528000020980835 simga_cut_n[89,4]
   + 0.6600000262260437 simga_cut_n[89,5]
   + 0.7920000314712524 simga_cut_n[89,6]
   + 0.9240000367164611 simga_cut_n[89,7]
   + 1.05600004196167 simga_cut_n[89,8]
   + 1.188000047206879 simga_cut_n[89,9] <= 1.320000052452087
 c_y_cut[90]: 0.1399999976158142 simga_cut_n[90,1]
   + 0.2799999952316284 simga_cut_n[90,2]
   + 0.4199999928474426 simga_cut_n[90,3]
   + 0.5599999904632569 simga_cut_n[90,4]
   + 0.699999988079071 simga_cut_n[90,5]
   + 0.8399999856948852 simga_cut_n[90,6]
   + 0.9799999833106995 simga_cut_n[90,7]
   + 1.119999980926514 simga_cut_n[90,8]
   + 1.259999978542328 simga_cut_n[90,9] <= 1.399999976158142
 c_y_cut[91]: 0.1399999976158142 simga_cut_n[91,1]
   + 0.2799999952316284 simga_cut_n[91,2]
   + 0.4199999928474426 simga_cut_n[91,3]
   + 0.5599999904632569 simga_cut_n[91,4]
   + 0.699999988079071 simga_cut_n[91,5]
   + 0.8399999856948852 simga_cut_n[91,6]
   + 0.9799999833106995 simga_cut_n[91,7]
   + 1.119999980926514 simga_cut_n[91,8]
   + 1.259999978542328 simga_cut_n[91,9] <= 1.399999976158142
 c_y_cut[92]: 0.172000002861023 simga_cut_n[92,1]
   + 0.3440000057220459 simga_cut_n[92,2]
   + 0.5160000085830688 simga_cut_n[92,3]
   + 0.6880000114440918 simga_cut_n[92,4]
   + 0.8600000143051147 simga_cut_n[92,5]
   + 1.032000017166138 simga_cut_n[92,6]
   + 1.204000020027161 simga_cut_n[92,7]
   + 1.376000022888184 simga_cut_n[92,8]
   + 1.548000025749207 simga_cut_n[92,9] <= 1.720000028610229
 c_y_cut[93]: 0.1669999957084656 simga_cut_n[93,1]
   + 0.3339999914169312 simga_cut_n[93,2]
   + 0.5009999871253967 simga_cut_n[93,3]
   + 0.6679999828338623 simga_cut_n[93,4]
   + 0.8349999785423279 simga_cut_n[93,5]
   + 1.001999974250793 simga_cut_n[93,6]
   + 1.168999969959259 simga_cut_n[93,7]
   + 1.335999965667725 simga_cut_n[93,8]
   + 1.50299996137619 simga_cut_n[93,9] <= 1.669999957084656
 c_y_cut[94]: 0.172000002861023 simga_cut_n[94,1]
   + 0.3440000057220459 simga_cut_n[94,2]
   + 0.5160000085830688 simga_cut_n[94,3]
   + 0.6880000114440918 simga_cut_n[94,4]
   + 0.8600000143051147 simga_cut_n[94,5]
   + 1.032000017166138 simga_cut_n[94,6]
   + 1.204000020027161 simga_cut_n[94,7]
   + 1.376000022888184 simga_cut_n[94,8]
   + 1.548000025749207 simga_cut_n[94,9] <= 1.720000028610229
 c_y_cut[95]: 0.172000002861023 simga_cut_n[95,1]
   + 0.3440000057220459 simga_cut_n[95,2]
   + 0.5160000085830688 simga_cut_n[95,3]
   + 0.6880000114440918 simga_cut_n[95,4]
   + 0.8600000143051147 simga_cut_n[95,5]
   + 1.032000017166138 simga_cut_n[95,6]
   + 1.204000020027161 simga_cut_n[95,7]
   + 1.376000022888184 simga_cut_n[95,8]
   + 1.548000025749207 simga_cut_n[95,9] <= 1.720000028610229
 c_y_add[0]: <= 0
 c_y_add[1]: <= 0
 c_y_add[2]: <= 0
 c_y_add[3]: <= 0
 c_y_add[4]: <= 0
 c_y_add[5]: <= 0
 c_y_add[6]: <= 0
 c_y_add[7]: <= 0
 c_y_add[8]: <= 0
 c_y_add[9]: <= 0
 c_y_add[10]: <= 0
 c_y_add[11]: <= 0
 c_y_add[12]: <= 0
 c_y_add[13]: <= 0
 c_y_add[14]: <= 0
 c_y_add[15]: <= 0
 c_y_add[16]: <= 0
 c_y_add[17]: <= 0
 c_y_add[18]: <= 0
 c_y_add[19]: <= 0
 c_y_add[20]: <= 0
 c_y_add[21]: <= 0
 c_y_add[22]: <= 0
 c_y_add[23]: <= 0
 c_y_add[24]: <= 0
 c_y_add[25]: <= 0
 c_y_add[26]: <= 0
 c_y_add[27]: <= 0
 c_y_add[28]: <= 0
 c_y_add[29]: <= 0
 c_y_add[30]: <= 0
 c_y_add[31]: <= 0
 c_y_add[32]: <= 0
 c_y_add[33]: <= 0
 c_y_add[34]: <= 0
 c_y_add[35]: 1.227763366699219 simga_add_n[35,1]
   + 2.455526733398437 simga_add_n[35,2]
   + 3.683290100097656 simga_add_n[35,3]
   + 4.911053466796875 simga_add_n[35,4]
   + 6.138816833496094 simga_add_n[35,5]
   + 7.366580200195313 simga_add_n[35,6]
   + 8.594343566894532 simga_add_n[35,7]
   + 9.82210693359375 simga_add_n[35,8]
   + 11.04987030029297 simga_add_n[35,9] <= 12.27763366699219
 c_y_add[36]: 4.313101196289063 simga_add_n[36,1]
   + 8.626202392578126 simga_add_n[36,2]
   + 12.93930358886719 simga_add_n[36,3]
   + 17.25240478515625 simga_add_n[36,4]
   + 21.56550598144531 simga_add_n[36,5]
   + 25.87860717773437 simga_add_n[36,6]
   + 30.19170837402344 simga_add_n[36,7]
   + 34.5048095703125 simga_add_n[36,8]
   + 38.81791076660156 simga_add_n[36,9] <= 43.13101196289063
 c_y_add[37]: 7.312644958496094 simga_add_n[37,1]
   + 14.62528991699219 simga_add_n[37,2]
   + 21.93793487548828 simga_add_n[37,3]
   + 29.25057983398438 simga_add_n[37,4]
   + 36.56322479248047 simga_add_n[37,5]
   + 43.87586975097656 simga_add_n[37,6]
   + 51.18851470947266 simga_add_n[37,7]
   + 58.50115966796875 simga_add_n[37,8]
   + 65.81380462646484 simga_add_n[37,9] <= 73.12644958496094
 c_y_add[38]: 9.112806087358942 simga_add_n[38,1]
   + 18.22561217471788 simga_add_n[38,2]
   + 27.33841826207682 simga_add_n[38,3]
   + 36.45122434943577 simga_add_n[38,4]
   + 45.56403043679471 simga_add_n[38,5]
   + 54.67683652415364 simga_add_n[38,6]
   + 63.78964261151259 simga_add_n[38,7]
   + 72.90244869887154 simga_add_n[38,8]
   + 82.01525478623049 simga_add_n[38,9] <= 91.12806087358942
 c_y_add[39]: 11.36428925308036 simga_add_n[39,1]
   + 22.72857850616072 simga_add_n[39,2]
   + 34.09286775924108 simga_add_n[39,3]
   + 45.45715701232144 simga_add_n[39,4]
   + 56.8214462654018 simga_add_n[39,5]
   + 68.18573551848216 simga_add_n[39,6]
   + 79.55002477156252 simga_add_n[39,7]
   + 90.91431402464288 simga_add_n[39,8]
   + 102.2786032777232 simga_add_n[39,9] <= 113.6428925308036
 c_y_add[40]: 12.36420123814019 simga_add_n[40,1]
   + 24.72840247628037 simga_add_n[40,2]
   + 37.09260371442056 simga_add_n[40,3]
   + 49.45680495256074 simga_add_n[40,4]
   + 61.82100619070094 simga_add_n[40,5]
   + 74.18520742884112 simga_add_n[40,6]
   + 86.54940866698131 simga_add_n[40,7]
   + 98.91360990512149 simga_add_n[40,8]
   + 111.2778111432617 simga_add_n[40,9] <= 123.6420123814019
 c_y_add[41]: 13.13428974705986 simga_add_n[41,1]
   + 26.26857949411972 simga_add_n[41,2]
   + 39.40286924117957 simga_add_n[41,3]
   + 52.53715898823943 simga_add_n[41,4]
   + 65.67144873529929 simga_add_n[41,5]
   + 78.80573848235915 simga_add_n[41,6]
   + 91.94002822941901 simga_add_n[41,7]
   + 105.0743179764789 simga_add_n[41,8]
   + 118.2086077235387 simga_add_n[41,9] <= 131.3428974705986
 c_y_add[42]: 13.25205560161239 simga_add_n[42,1]
   + 26.50411120322478 simga_add_n[42,2]
   + 39.75616680483718 simga_add_n[42,3]
   + 53.00822240644957 simga_add_n[42,4]
   + 66.26027800806196 simga_add_n[42,5]
   + 79.51233360967436 simga_add_n[42,6]
   + 92.76438921128674 simga_add_n[42,7]
   + 106.0164448128991 simga_add_n[42,8]
   + 119.2685004145115 simga_add_n[42,9] <= 132.5205560161239
 c_y_add[43]: 17.76337805778585 simga_add_n[43,1]
   + 35.5267561155717 simga_add_n[43,2]
   + 53.29013417335754 simga_add_n[43,3]
   + 71.0535122311434 simga_add_n[43,4]
   + 88.81689028892924 simga_add_n[43,5]
   + 106.5802683467151 simga_add_n[43,6]
   + 124.3436464045009 simga_add_n[43,7]
   + 142.1070244622868 simga_add_n[43,8]
   + 159.8704025200726 simga_add_n[43,9] <= 177.6337805778585
 c_y_add[44]: 17.91483213285851 simga_add_n[44,1]
   + 35.82966426571702 simga_add_n[44,2]
   + 53.74449639857553 simga_add_n[44,3]
   + 71.65932853143404 simga_add_n[44,4]
   + 89.57416066429255 simga_add_n[44,5]
   + 107.4889927971511 simga_add_n[44,6]
   + 125.4038249300096 simga_add_n[44,7]
   + 143.3186570628681 simga_add_n[44,8]
   + 161.2334891957266 simga_add_n[44,9] <= 179.1483213285851
 c_y_add[45]: 18.47520235360185 simga_add_n[45,1]
   + 36.9504047072037 simga_add_n[45,2]
   + 55.42560706080555 simga_add_n[45,3]
   + 73.90080941440741 simga_add_n[45,4]
   + 92.37601176800926 simga_add_n[45,5]
   + 110.8512141216111 simga_add_n[45,6]
   + 129.326416475213 simga_add_n[45,7]
   + 147.8016188288148 simga_add_n[45,8]
   + 166.2768211824167 simga_add_n[45,9] <= 184.7520235360185
 c_y_add[46]: 15.84882492159153 simga_add_n[46,1]
   + 31.69764984318306 simga_add_n[46,2]
   + 47.54647476477459 simga_add_n[46,3]
   + 63.39529968636612 simga_add_n[46,4]
   + 79.24412460795764 simga_add_n[46,5]
   + 95.09294952954917 simga_add_n[46,6]
   + 110.9417744511407 simga_add_n[46,7]
   + 126.7905993727322 simga_add_n[46,8]
   + 142.6394242943238 simga_add_n[46,9] <= 158.4882492159153
 c_y_add[47]: 16.08938811955057 simga_add_n[47,1]
   + 32.17877623910115 simga_add_n[47,2]
   + 48.26816435865172 simga_add_n[47,3]
   + 64.35755247820229 simga_add_n[47,4]
   + 80.44694059775287 simga_add_n[47,5]
   + 96.53632871730345 simga_add_n[47,6]
   + 112.625716836854 simga_add_n[47,7]
   + 128.7151049564046 simga_add_n[47,8]
   + 144.8044930759552 simga_add_n[47,9] <= 160.8938811955057
 c_y_add[48]: 15.81114578286554 simga_add_n[48,1]
   + 31.62229156573108 simga_add_n[48,2]
   + 47.43343734859661 simga_add_n[48,3]
   + 63.24458313146215 simga_add_n[48,4]
   + 79.05572891432769 simga_add_n[48,5]
   + 94.86687469719323 simga_add_n[48,6]
   + 110.6780204800588 simga_add_n[48,7]
   + 126.4891662629243 simga_add_n[48,8]
   + 142.3003120457898 simga_add_n[48,9] <= 158.1114578286554
 c_y_add[49]: 16.74851059963887 simga_add_n[49,1]
   + 33.49702119927773 simga_add_n[49,2]
   + 50.24553179891659 simga_add_n[49,3]
   + 66.99404239855546 simga_add_n[49,4]
   + 83.74255299819433 simga_add_n[49,5]
   + 100.4910635978332 simga_add_n[49,6]
   + 117.2395741974721 simga_add_n[49,7]
   + 133.9880847971109 simga_add_n[49,8]
   + 150.7365953967498 simga_add_n[49,9] <= 167.4851059963887
 c_y_add[50]: 15.88006948903661 simga_add_n[50,1]
   + 31.76013897807322 simga_add_n[50,2]
   + 47.64020846710984 simga_add_n[50,3]
   + 63.52027795614644 simga_add_n[50,4]
   + 79.40034744518306 simga_add_n[50,5]
   + 95.28041693421967 simga_add_n[50,6]
   + 111.1604864232563 simga_add_n[50,7]
   + 127.0405559122929 simga_add_n[50,8]
   + 142.9206254013295 simga_add_n[50,9] <= 158.8006948903661
 c_y_add[51]: 16.11441022098263 simga_add_n[51,1]
   + 32.22882044196527 simga_add_n[51,2]
   + 48.3432306629479 simga_add_n[51,3]
   + 64.45764088393054 simga_add_n[51,4]
   + 80.57205110491317 simga_add_n[51,5]
   + 96.68646132589581 simga_add_n[51,6]
   + 112.8008715468784 simga_add_n[51,7]
   + 128.9152817678611 simga_add_n[51,8]
   + 145.0296919888437 simga_add_n[51,9] <= 161.1441022098263
 c_y_add[52]: 15.76581025964014 simga_add_n[52,1]
   + 31.53162051928028 simga_add_n[52,2]
   + 47.29743077892043 simga_add_n[52,3]
   + 63.06324103856057 simga_add_n[52,4]
   + 78.82905129820071 simga_add_n[52,5]
   + 94.59486155784086 simga_add_n[52,6]
   + 110.360671817481 simga_add_n[52,7]
   + 126.1264820771211 simga_add_n[52,8]
   + 141.8922923367613 simga_add_n[52,9] <= 157.6581025964014
 c_y_add[53]: 15.61591886327097 simga_add_n[53,1]
   + 31.23183772654193 simga_add_n[53,2]
   + 46.8477565898129 simga_add_n[53,3]
   + 62.46367545308387 simga_add_n[53,4]
   + 78.07959431635483 simga_add_n[53,5]
   + 93.69551317962581 simga_add_n[53,6]
   + 109.3114320428968 simga_add_n[53,7]
   + 124.9273509061677 simga_add_n[53,8]
   + 140.5432697694387 simga_add_n[53,9] <= 156.1591886327097
 c_y_add[54]: 15.91570043536703 simga_add_n[54,1]
   + 31.83140087073406 simga_add_n[54,2]
   + 47.74710130610109 simga_add_n[54,3]
   + 63.66280174146812 simga_add_n[54,4]
   + 79.57850217683514 simga_add_n[54,5]
   + 95.49420261220217 simga_add_n[54,6]
   + 111.4099030475692 simga_add_n[54,7]
   + 127.3256034829362 simga_add_n[54,8]
   + 143.2413039183033 simga_add_n[54,9] <= 159.1570043536703
 c_y_add[55]: 16.1925723833438 simga_add_n[55,1]
   + 32.3851447666876 simga_add_n[55,2]
   + 48.5777171500314 simga_add_n[55,3]
   + 64.77028953337519 simga_add_n[55,4]
   + 80.96286191671899 simga_add_n[55,5]
   + 97.15543430006279 simga_add_n[55,6]
   + 113.3480066834066 simga_add_n[55,7]
   + 129.5405790667504 simga_add_n[55,8]
   + 145.7331514500942 simga_add_n[55,9] <= 161.925723833438
 c_y_add[56]: 17.03567184676319 simga_add_n[56,1]
   + 34.07134369352639 simga_add_n[56,2]
   + 51.10701554028958 simga_add_n[56,3]
   + 68.14268738705277 simga_add_n[56,4]
   + 85.17835923381597 simga_add_n[56,5]
   + 102.2140310805792 simga_add_n[56,6]
   + 119.2497029273424 simga_add_n[56,7]
   + 136.2853747741055 simga_add_n[56,8]
   + 153.3210466208687 simga_add_n[56,9] <= 170.3567184676319
 c_y_add[57]: 17.50933868555562 simga_add_n[57,1]
   + 35.01867737111125 simga_add_n[57,2]
   + 52.52801605666686 simga_add_n[57,3]
   + 70.03735474222249 simga_add_n[57,4]
   + 87.54669342777811 simga_add_n[57,5]
   + 105.0560321133337 simga_add_n[57,6]
   + 122.5653707988893 simga_add_n[57,7]
   + 140.074709484445 simga_add_n[57,8]
   + 157.5840481700006 simga_add_n[57,9] <= 175.0933868555562
 c_y_add[58]: 17.21533902039349 simga_add_n[58,1]
   + 34.43067804078699 simga_add_n[58,2]
   + 51.64601706118049 simga_add_n[58,3]
   + 68.86135608157397 simga_add_n[58,4]
   + 86.07669510196747 simga_add_n[58,5]
   + 103.292034122361 simga_add_n[58,6]
   + 120.5073731427545 simga_add_n[58,7]
   + 137.7227121631479 simga_add_n[58,8]
   + 154.9380511835415 simga_add_n[58,9] <= 172.1533902039349
 c_y_add[59]: 18.48933915222707 simga_add_n[59,1]
   + 36.97867830445414 simga_add_n[59,2]
   + 55.46801745668121 simga_add_n[59,3]
   + 73.95735660890827 simga_add_n[59,4]
   + 92.44669576113535 simga_add_n[59,5]
   + 110.9360349133624 simga_add_n[59,6]
   + 129.4253740655895 simga_add_n[59,7]
   + 147.9147132178165 simga_add_n[59,8]
   + 166.4040523700436 simga_add_n[59,9] <= 184.8933915222707
 c_y_add[60]: 17.16977785362874 simga_add_n[60,1]
   + 34.33955570725749 simga_add_n[60,2]
   + 51.50933356088624 simga_add_n[60,3]
   + 68.67911141451498 simga_add_n[60,4]
   + 85.84888926814372 simga_add_n[60,5]
   + 103.0186671217725 simga_add_n[60,6]
   + 120.1884449754012 simga_add_n[60,7]
   + 137.35822282903 simga_add_n[60,8]
   + 154.5280006826587 simga_add_n[60,9] <= 171.6977785362874
 c_y_add[61]: 15.9845205425066 simga_add_n[61,1]
   + 31.9690410850132 simga_add_n[61,2]
   + 47.9535616275198 simga_add_n[61,3]
   + 63.93808217002639 simga_add_n[61,4]
   + 79.92260271253299 simga_add_n[61,5]
   + 95.90712325503959 simga_add_n[61,6]
   + 111.8916437975462 simga_add_n[61,7]
   + 127.8761643400528 simga_add_n[61,8]
   + 143.8606848825594 simga_add_n[61,9] <= 159.845205425066
 c_y_add[62]: 16.84328396738097 simga_add_n[62,1]
   + 33.68656793476195 simga_add_n[62,2]
   + 50.52985190214292 simga_add_n[62,3]
   + 67.37313586952389 simga_add_n[62,4]
   + 84.21641983690486 simga_add_n[62,5]
   + 101.0597038042858 simga_add_n[62,6]
   + 117.9029877716668 simga_add_n[62,7]
   + 134.7462717390478 simga_add_n[62,8]
   + 151.5895557064287 simga_add_n[62,9] <= 168.4328396738097
 c_y_add[63]: 14.71586475535454 simga_add_n[63,1]
   + 29.43172951070908 simga_add_n[63,2]
   + 44.14759426606362 simga_add_n[63,3]
   + 58.86345902141816 simga_add_n[63,4]
   + 73.5793237767727 simga_add_n[63,5]
   + 88.29518853212724 simga_add_n[63,6]
   + 103.0110532874818 simga_add_n[63,7]
   + 117.7269180428363 simga_add_n[63,8]
   + 132.4427827981909 simga_add_n[63,9] <= 147.1586475535454
 c_y_add[64]: 10.83350698647833 simga_add_n[64,1]
   + 21.66701397295666 simga_add_n[64,2]
   + 32.50052095943499 simga_add_n[64,3]
   + 43.33402794591333 simga_add_n[64,4]
   + 54.16753493239166 simga_add_n[64,5]
   + 65.00104191886999 simga_add_n[64,6]
   + 75.83454890534833 simga_add_n[64,7]
   + 86.66805589182665 simga_add_n[64,8]
   + 97.50156287830499 simga_add_n[64,9] <= 108.3350698647833
 c_y_add[65]: 8.898127099008061 simga_add_n[65,1]
   + 17.79625419801612 simga_add_n[65,2]
   + 26.69438129702418 simga_add_n[65,3]
   + 35.59250839603224 simga_add_n[65,4]
   + 44.4906354950403 simga_add_n[65,5]
   + 53.38876259404837 simga_add_n[65,6]
   + 62.28688969305642 simga_add_n[65,7]
   + 71.18501679206449 simga_add_n[65,8]
   + 80.08314389107254 simga_add_n[65,9] <= 88.9812709900806
 c_y_add[66]: 1.026481883702014 simga_add_n[66,1]
   + 2.052963767404029 simga_add_n[66,2]
   + 3.079445651106043 simga_add_n[66,3]
   + 4.105927534808058 simga_add_n[66,4]
   + 5.132409418510072 simga_add_n[66,5]
   + 6.158891302212086 simga_add_n[66,6]
   + 7.1853731859141 simga_add_n[66,7]
   + 8.211855069616115 simga_add_n[66,8]
   + 9.238336953318129 simga_add_n[66,9] <= 10.26481883702014
 c_y_add[67]: 0.6802087703268661 simga_add_n[67,1]
   + 1.360417540653732 simga_add_n[67,2]
   + 2.040626310980598 simga_add_n[67,3]
   + 2.720835081307464 simga_add_n[67,4]
   + 3.401043851634331 simga_add_n[67,5]
   + 4.081252621961196 simga_add_n[67,6]
   + 4.761461392288063 simga_add_n[67,7]
   + 5.441670162614929 simga_add_n[67,8]
   + 6.121878932941796 simga_add_n[67,9] <= 6.802087703268661
 c_y_add[68]: <= 0
 c_y_add[69]: <= 0
 c_y_add[70]: <= 0
 c_y_add[71]: <= 0
 c_y_add[72]: <= 0
 c_y_add[73]: <= 0
 c_y_add[74]: <= 0
 c_y_add[75]: <= 0
 c_y_add[76]: <= 0
 c_y_add[77]: <= 0
 c_y_add[78]: <= 0
 c_y_add[79]: <= 0
 c_y_add[80]: <= 0
 c_y_add[81]: <= 0
 c_y_add[82]: <= 0
 c_y_add[83]: <= 0
 c_y_add[84]: <= 0
 c_y_add[85]: <= 0
 c_y_add[86]: <= 0
 c_y_add[87]: <= 0
 c_y_add[88]: <= 0
 c_y_add[89]: <= 0
 c_y_add[90]: <= 0
 c_y_add[91]: <= 0
 c_y_add[92]: <= 0
 c_y_add[93]: <= 0
 c_y_add[94]: <= 0
 c_y_add[95]: <= 0
 c_power_balance_eq[0]: y_pos[0] - y_neg[0] - y_chg[0] + y_dis[0] + y_PV[0]
   - y_load[0] - 0.1690000057220459 simga_cut_n[0,1]
   - 0.3380000114440918 simga_cut_n[0,2]
   - 0.5070000171661377 simga_cut_n[0,3]
   - 0.6760000228881836 simga_cut_n[0,4]
   - 0.8450000286102295 simga_cut_n[0,5]
   - 1.014000034332275 simga_cut_n[0,6]
   - 1.183000040054321 simga_cut_n[0,7]
   - 1.352000045776367 simga_cut_n[0,8]
   - 1.521000051498413 simga_cut_n[0,9] = -306.4650883674622
 c_power_balance_eq[1]: y_pos[1] - y_neg[1] - y_chg[1] + y_dis[1] + y_PV[1]
   - y_load[1] - 0.1690000057220459 simga_cut_n[1,1]
   - 0.3380000114440918 simga_cut_n[1,2]
   - 0.5070000171661377 simga_cut_n[1,3]
   - 0.6760000228881836 simga_cut_n[1,4]
   - 0.8450000286102295 simga_cut_n[1,5]
   - 1.014000034332275 simga_cut_n[1,6]
   - 1.183000040054321 simga_cut_n[1,7]
   - 1.352000045776367 simga_cut_n[1,8]
   - 1.521000051498413 simga_cut_n[1,9] = -304.735077381134
 c_power_balance_eq[2]: y_pos[2] - y_neg[2] - y_chg[2] + y_dis[2] + y_PV[2]
   - y_load[2] - 0.1700000047683716 simga_cut_n[2,1]
   - 0.3400000095367431 simga_cut_n[2,2]
   - 0.5100000143051148 simga_cut_n[2,3]
   - 0.6800000190734863 simga_cut_n[2,4]
   - 0.8500000238418579 simga_cut_n[2,5]
   - 1.02000002861023 simga_cut_n[2,6] - 1.190000033378601 simga_cut_n[2,7]
   - 1.360000038146973 simga_cut_n[2,8]
   - 1.530000042915344 simga_cut_n[2,9] = -307.3750920295715
 c_power_balance_eq[3]: y_pos[3] - y_neg[3] - y_chg[3] + y_dis[3] + y_PV[3]
   - y_load[3] - 0.1700000047683716 simga_cut_n[3,1]
   - 0.3400000095367431 simga_cut_n[3,2]
   - 0.5100000143051148 simga_cut_n[3,3]
   - 0.6800000190734863 simga_cut_n[3,4]
   - 0.8500000238418579 simga_cut_n[3,5]
   - 1.02000002861023 simga_cut_n[3,6] - 1.190000033378601 simga_cut_n[3,7]
   - 1.360000038146973 simga_cut_n[3,8]
   - 1.530000042915344 simga_cut_n[3,9] = -306.1250920295715
 c_power_balance_eq[4]: y_pos[4] - y_neg[4] - y_chg[4] + y_dis[4] + y_PV[4]
   - y_load[4] - 0.1700000047683716 simga_cut_n[4,1]
   - 0.3400000095367431 simga_cut_n[4,2]
   - 0.5100000143051148 simga_cut_n[4,3]
   - 0.6800000190734863 simga_cut_n[4,4]
   - 0.8500000238418579 simga_cut_n[4,5]
   - 1.02000002861023 simga_cut_n[4,6] - 1.190000033378601 simga_cut_n[4,7]
   - 1.360000038146973 simga_cut_n[4,8]
   - 1.530000042915344 simga_cut_n[4,9] = -306.4650883674622
 c_power_balance_eq[5]: y_pos[5] - y_neg[5] - y_chg[5] + y_dis[5] + y_PV[5]
   - y_load[5] - 0.2039999961853027 simga_cut_n[5,1]
   - 0.4079999923706055 simga_cut_n[5,2]
   - 0.6119999885559082 simga_cut_n[5,3]
   - 0.8159999847412109 simga_cut_n[5,4]
   - 1.019999980926514 simga_cut_n[5,5]
   - 1.223999977111816 simga_cut_n[5,6]
   - 1.427999973297119 simga_cut_n[5,7]
   - 1.631999969482422 simga_cut_n[5,8]
   - 1.835999965667725 simga_cut_n[5,9] = -308.3861150741577
 c_power_balance_eq[6]: y_pos[6] - y_neg[6] - y_chg[6] + y_dis[6] + y_PV[6]
   - y_load[6] - 0.2039999961853027 simga_cut_n[6,1]
   - 0.4079999923706055 simga_cut_n[6,2]
   - 0.6119999885559082 simga_cut_n[6,3]
   - 0.8159999847412109 simga_cut_n[6,4]
   - 1.019999980926514 simga_cut_n[6,5]
   - 1.223999977111816 simga_cut_n[6,6]
   - 1.427999973297119 simga_cut_n[6,7]
   - 1.631999969482422 simga_cut_n[6,8]
   - 1.835999965667725 simga_cut_n[6,9] = -304.3461065292358
 c_power_balance_eq[7]: y_pos[7] - y_neg[7] - y_chg[7] + y_dis[7] + y_PV[7]
   - y_load[7] - 0.2039999961853027 simga_cut_n[7,1]
   - 0.4079999923706055 simga_cut_n[7,2]
   - 0.6119999885559082 simga_cut_n[7,3]
   - 0.8159999847412109 simga_cut_n[7,4]
   - 1.019999980926514 simga_cut_n[7,5]
   - 1.223999977111816 simga_cut_n[7,6]
   - 1.427999973297119 simga_cut_n[7,7]
   - 1.631999969482422 simga_cut_n[7,8]
   - 1.835999965667725 simga_cut_n[7,9] = -307.9061040878296
 c_power_balance_eq[8]: y_pos[8] - y_neg[8] - y_chg[8] + y_dis[8] + y_PV[8]
   - y_load[8] - 0.2039999961853027 simga_cut_n[8,1]
   - 0.4079999923706055 simga_cut_n[8,2]
   - 0.6119999885559082 simga_cut_n[8,3]
   - 0.8159999847412109 simga_cut_n[8,4]
   - 1.019999980926514 simga_cut_n[8,5]
   - 1.223999977111816 simga_cut_n[8,6]
   - 1.427999973297119 simga_cut_n[8,7]
   - 1.631999969482422 simga_cut_n[8,8]
   - 1.835999965667725 simga_cut_n[8,9] = -306.3660955429077
 c_power_balance_eq[9]: y_pos[9] - y_neg[9] - y_chg[9] + y_dis[9] + y_PV[9]
   - y_load[9] - 0.2039999961853027 simga_cut_n[9,1]
   - 0.4079999923706055 simga_cut_n[9,2]
   - 0.6119999885559082 simga_cut_n[9,3]
   - 0.8159999847412109 simga_cut_n[9,4]
   - 1.019999980926514 simga_cut_n[9,5]
   - 1.223999977111816 simga_cut_n[9,6]
   - 1.427999973297119 simga_cut_n[9,7]
   - 1.631999969482422 simga_cut_n[9,8]
   - 1.835999965667725 simga_cut_n[9,9] = -307.4661016464233
 c_power_balance_eq[10]: y_pos[10] - y_neg[10] - y_chg[10] + y_dis[10]
   + y_PV[10] - y_load[10] - 0.2380000114440918 simga_cut_n[10,1]
   - 0.4760000228881836 simga_cut_n[10,2]
   - 0.7140000343322754 simga_cut_n[10,3]
   - 0.9520000457763672 simga_cut_n[10,4]
   - 1.190000057220459 simga_cut_n[10,5]
   - 1.428000068664551 simga_cut_n[10,6]
   - 1.666000080108643 simga_cut_n[10,7]
   - 1.904000091552734 simga_cut_n[10,8]
   - 2.142000102996826 simga_cut_n[10,9] = -308.2871222496033
 c_power_balance_eq[11]: y_pos[11] - y_neg[11] - y_chg[11] + y_dis[11]
   + y_PV[11] - y_load[11] - 0.2380000114440918 simga_cut_n[11,1]
   - 0.4760000228881836 simga_cut_n[11,2]
   - 0.7140000343322754 simga_cut_n[11,3]
   - 0.9520000457763672 simga_cut_n[11,4]
   - 1.190000057220459 simga_cut_n[11,5]
   - 1.428000068664551 simga_cut_n[11,6]
   - 1.666000080108643 simga_cut_n[11,7]
   - 1.904000091552734 simga_cut_n[11,8]
   - 2.142000102996826 simga_cut_n[11,9] = -307.4671149253845
 c_power_balance_eq[12]: y_pos[12] - y_neg[12] - y_chg[12] + y_dis[12]
   + y_PV[12] - y_load[12] - 0.2369999885559082 simga_cut_n[12,1]
   - 0.4739999771118164 simga_cut_n[12,2]
   - 0.7109999656677246 simga_cut_n[12,3]
   - 0.9479999542236328 simga_cut_n[12,4]
   - 1.184999942779541 simga_cut_n[12,5]
   - 1.421999931335449 simga_cut_n[12,6]
   - 1.658999919891357 simga_cut_n[12,7]
   - 1.895999908447266 simga_cut_n[12,8]
   - 2.132999897003174 simga_cut_n[12,9] = -306.6471076011658
 c_power_balance_eq[13]: y_pos[13] - y_neg[13] - y_chg[13] + y_dis[13]
   + y_PV[13] - y_load[13] - 0.2369999885559082 simga_cut_n[13,1]
   - 0.4739999771118164 simga_cut_n[13,2]
   - 0.7109999656677246 simga_cut_n[13,3]
   - 0.9479999542236328 simga_cut_n[13,4]
   - 1.184999942779541 simga_cut_n[13,5]
   - 1.421999931335449 simga_cut_n[13,6]
   - 1.658999919891357 simga_cut_n[13,7]
   - 1.895999908447266 simga_cut_n[13,8]
   - 2.132999897003174 simga_cut_n[13,9] = -309.2871222496033
 c_power_balance_eq[14]: y_pos[14] - y_neg[14] - y_chg[14] + y_dis[14]
   + y_PV[14] - y_load[14] - 0.2369999885559082 simga_cut_n[14,1]
   - 0.4739999771118164 simga_cut_n[14,2]
   - 0.7109999656677246 simga_cut_n[14,3]
   - 0.9479999542236328 simga_cut_n[14,4]
   - 1.184999942779541 simga_cut_n[14,5]
   - 1.421999931335449 simga_cut_n[14,6]
   - 1.658999919891357 simga_cut_n[14,7]
   - 1.895999908447266 simga_cut_n[14,8]
   - 2.132999897003174 simga_cut_n[14,9] = -305.7371039390564
 c_power_balance_eq[15]: y_pos[15] - y_neg[15] - y_chg[15] + y_dis[15]
   + y_PV[15] - y_load[15] - 0.2710000038146972 simga_cut_n[15,1]
   - 0.5420000076293945 simga_cut_n[15,2]
   - 0.8130000114440918 simga_cut_n[15,3]
   - 1.084000015258789 simga_cut_n[15,4]
   - 1.355000019073486 simga_cut_n[15,5]
   - 1.626000022888184 simga_cut_n[15,6]
   - 1.897000026702881 simga_cut_n[15,7]
   - 2.168000030517578 simga_cut_n[15,8]
   - 2.439000034332275 simga_cut_n[15,9] = -309.2381477355957
 c_power_balance_eq[16]: y_pos[16] - y_neg[16] - y_chg[16] + y_dis[16]
   + y_PV[16] - y_load[16] - 0.2710000038146972 simga_cut_n[16,1]
   - 0.5420000076293945 simga_cut_n[16,2]
   - 0.8130000114440918 simga_cut_n[16,3]
   - 1.084000015258789 simga_cut_n[16,4]
   - 1.355000019073486 simga_cut_n[16,5]
   - 1.626000022888184 simga_cut_n[16,6]
   - 1.897000026702881 simga_cut_n[16,7]
   - 2.168000030517578 simga_cut_n[16,8]
   - 2.439000034332275 simga_cut_n[16,9] = -304.2981452941895
 c_power_balance_eq[17]: y_pos[17] - y_neg[17] - y_chg[17] + y_dis[17]
   + y_PV[17] - y_load[17] - 0.2710000038146972 simga_cut_n[17,1]
   - 0.5420000076293945 simga_cut_n[17,2]
   - 0.8130000114440918 simga_cut_n[17,3]
   - 1.084000015258789 simga_cut_n[17,4]
   - 1.355000019073486 simga_cut_n[17,5]
   - 1.626000022888184 simga_cut_n[17,6]
   - 1.897000026702881 simga_cut_n[17,7]
   - 2.168000030517578 simga_cut_n[17,8]
   - 2.439000034332275 simga_cut_n[17,9] = -306.2681465148926
 c_power_balance_eq[18]: y_pos[18] - y_neg[18] - y_chg[18] + y_dis[18]
   + y_PV[18] - y_load[18] - 0.2710000038146972 simga_cut_n[18,1]
   - 0.5420000076293945 simga_cut_n[18,2]
   - 0.8130000114440918 simga_cut_n[18,3]
   - 1.084000015258789 simga_cut_n[18,4]
   - 1.355000019073486 simga_cut_n[18,5]
   - 1.626000022888184 simga_cut_n[18,6]
   - 1.897000026702881 simga_cut_n[18,7]
   - 2.168000030517578 simga_cut_n[18,8]
   - 2.439000034332275 simga_cut_n[18,9] = -305.4481391906738
 c_power_balance_eq[19]: y_pos[19] - y_neg[19] - y_chg[19] + y_dis[19]
   + y_PV[19] - y_load[19] - 0.2710000038146972 simga_cut_n[19,1]
   - 0.5420000076293945 simga_cut_n[19,2]
   - 0.8130000114440918 simga_cut_n[19,3]
   - 1.084000015258789 simga_cut_n[19,4]
   - 1.355000019073486 simga_cut_n[19,5]
   - 1.626000022888184 simga_cut_n[19,6]
   - 1.897000026702881 simga_cut_n[19,7]
   - 2.168000030517578 simga_cut_n[19,8]
   - 2.439000034332275 simga_cut_n[19,9] = -304.918140411377
 c_power_balance_eq[20]: y_pos[20] - y_neg[20] - y_chg[20] + y_dis[20]
   + y_PV[20] - y_load[20] - 0.3049999952316284 simga_cut_n[20,1]
   - 0.6099999904632568 simga_cut_n[20,2]
   - 0.9149999856948853 simga_cut_n[20,3]
   - 1.219999980926514 simga_cut_n[20,4]
   - 1.524999976158142 simga_cut_n[20,5]
   - 1.829999971389771 simga_cut_n[20,6]
   - 2.134999966621399 simga_cut_n[20,7]
   - 2.439999961853027 simga_cut_n[20,8]
   - 2.744999957084656 simga_cut_n[20,9] = -305.7291512489319
 c_power_balance_eq[21]: y_pos[21] - y_neg[21] - y_chg[21] + y_dis[21]
   + y_PV[21] - y_load[21] - 0.3049999952316284 simga_cut_n[21,1]
   - 0.6099999904632568 simga_cut_n[21,2]
   - 0.9149999856948853 simga_cut_n[21,3]
   - 1.219999980926514 simga_cut_n[21,4]
   - 1.524999976158142 simga_cut_n[21,5]
   - 1.829999971389771 simga_cut_n[21,6]
   - 2.134999966621399 simga_cut_n[21,7]
   - 2.439999961853027 simga_cut_n[21,8]
   - 2.744999957084656 simga_cut_n[21,9] = -304.3891549110413
 c_power_balance_eq[22]: y_pos[22] - y_neg[22] - y_chg[22] + y_dis[22]
   + y_PV[22] - y_load[22] - 0.3049999952316284 simga_cut_n[22,1]
   - 0.6099999904632568 simga_cut_n[22,2]
   - 0.9149999856948853 simga_cut_n[22,3]
   - 1.219999980926514 simga_cut_n[22,4]
   - 1.524999976158142 simga_cut_n[22,5]
   - 1.829999971389771 simga_cut_n[22,6]
   - 2.134999966621399 simga_cut_n[22,7]
   - 2.439999961853027 simga_cut_n[22,8]
   - 2.744999957084656 simga_cut_n[22,9] = -304.9691414833069
 c_power_balance_eq[23]: y_pos[23] - y_neg[23] - y_chg[23] + y_dis[23]
   + y_PV[23] - y_load[23] - 0.3049999952316284 simga_cut_n[23,1]
   - 0.6099999904632568 simga_cut_n[23,2]
   - 0.9149999856948853 simga_cut_n[23,3]
   - 1.219999980926514 simga_cut_n[23,4]
   - 1.524999976158142 simga_cut_n[23,5]
   - 1.829999971389771 simga_cut_n[23,6]
   - 2.134999966621399 simga_cut_n[23,7]
   - 2.439999961853027 simga_cut_n[23,8]
   - 2.744999957084656 simga_cut_n[23,9] = -306.0191597938538
 c_power_balance_eq[24]: y_pos[24] - y_neg[24] - y_chg[24] + y_dis[24]
   + y_PV[24] - y_load[24] - 0.3049999952316284 simga_cut_n[24,1]
   - 0.6099999904632568 simga_cut_n[24,2]
   - 0.9149999856948853 simga_cut_n[24,3]
   - 1.219999980926514 simga_cut_n[24,4]
   - 1.524999976158142 simga_cut_n[24,5]
   - 1.829999971389771 simga_cut_n[24,6]
   - 2.134999966621399 simga_cut_n[24,7]
   - 2.439999961853027 simga_cut_n[24,8]
   - 2.744999957084656 simga_cut_n[24,9] = -303.4791512489319
 c_power_balance_eq[25]: y_pos[25] - y_neg[25] - y_chg[25] + y_dis[25]
   + y_PV[25] - y_load[25] - 0.3390000104904175 simga_cut_n[25,1]
   - 0.678000020980835 simga_cut_n[25,2]
   - 1.017000031471252 simga_cut_n[25,3]
   - 1.35600004196167 simga_cut_n[25,4]
   - 1.695000052452087 simga_cut_n[25,5]
   - 2.034000062942505 simga_cut_n[25,6]
   - 2.373000073432922 simga_cut_n[25,7]
   - 2.71200008392334 simga_cut_n[25,8]
   - 3.051000094413757 simga_cut_n[25,9] = -305.5401620864868
 c_power_balance_eq[26]: y_pos[26] - y_neg[26] - y_chg[26] + y_dis[26]
   + y_PV[26] - y_load[26] - 0.3049999952316284 simga_cut_n[26,1]
   - 0.6099999904632568 simga_cut_n[26,2]
   - 0.9149999856948853 simga_cut_n[26,3]
   - 1.219999980926514 simga_cut_n[26,4]
   - 1.524999976158142 simga_cut_n[26,5]
   - 1.829999971389771 simga_cut_n[26,6]
   - 2.134999966621399 simga_cut_n[26,7]
   - 2.439999961853027 simga_cut_n[26,8]
   - 2.744999957084656 simga_cut_n[26,9] = -302.5191597938538
 c_power_balance_eq[27]: y_pos[27] - y_neg[27] - y_chg[27] + y_dis[27]
   + y_PV[27] - y_load[27] - 1.119999980926514 simga_cut_n[27,1]
   - 2.239999961853028 simga_cut_n[27,2]
   - 3.359999942779541 simga_cut_n[27,3]
   - 4.479999923706055 simga_cut_n[27,4]
   - 5.599999904632568 simga_cut_n[27,5]
   - 6.719999885559082 simga_cut_n[27,6]
   - 7.839999866485596 simga_cut_n[27,7]
   - 8.95999984741211 simga_cut_n[27,8]
   - 10.07999982833862 simga_cut_n[27,9] = -294.5235509872437
 c_power_balance_eq[28]: y_pos[28] - y_neg[28] - y_chg[28] + y_dis[28]
   + y_PV[28] - y_load[28] - 2.025 simga_cut_n[28,1]
   - 4.05 simga_cut_n[28,2] - 6.075 simga_cut_n[28,3]
   - 8.1 simga_cut_n[28,4] - 10.125 simga_cut_n[28,5]
   - 12.15 simga_cut_n[28,6] - 14.175 simga_cut_n[28,7]
   - 16.2 simga_cut_n[28,8] - 18.225 simga_cut_n[28,9]
   = -288.3999938964844
 c_power_balance_eq[29]: y_pos[29] - y_neg[29] - y_chg[29] + y_dis[29]
   + y_PV[29] - y_load[29] - 3.22599983215332 simga_cut_n[29,1]
   - 6.451999664306641 simga_cut_n[29,2]
   - 9.677999496459961 simga_cut_n[29,3]
   - 12.90399932861328 simga_cut_n[29,4]
   - 16.1299991607666 simga_cut_n[29,5]
   - 19.35599899291992 simga_cut_n[29,6]
   - 22.58199882507324 simga_cut_n[29,7]
   - 25.80799865722656 simga_cut_n[29,8]
   - 29.03399848937988 simga_cut_n[29,9] = -273.0755958557129
 c_power_balance_eq[30]: y_pos[30] - y_neg[30] - y_chg[30] + y_dis[30]
   + y_PV[30] - y_load[30] - 2.403000068664551 simga_cut_n[30,1]
   - 4.806000137329102 simga_cut_n[30,2]
   - 7.209000205993652 simga_cut_n[30,3]
   - 9.612000274658204 simga_cut_n[30,4]
   - 12.01500034332275 simga_cut_n[30,5]
   - 14.4180004119873 simga_cut_n[30,6]
   - 16.82100048065185 simga_cut_n[30,7]
   - 19.22400054931641 simga_cut_n[30,8]
   - 21.62700061798096 simga_cut_n[30,9] = -282.6111831665039
 c_power_balance_eq[31]: y_pos[31] - y_neg[31] - y_chg[31] + y_dis[31]
   + y_PV[31] - y_load[31] - 6.109000015258789 simga_cut_n[31,1]
   - 12.21800003051758 simga_cut_n[31,2]
   - 18.32700004577637 simga_cut_n[31,3]
   - 24.43600006103516 simga_cut_n[31,4]
   - 30.54500007629395 simga_cut_n[31,5]
   - 36.65400009155273 simga_cut_n[31,6]
   - 42.76300010681152 simga_cut_n[31,7]
   - 48.87200012207031 simga_cut_n[31,8]
   - 54.9810001373291 simga_cut_n[31,9] = -243.5010147094727
 c_power_balance_eq[32]: y_pos[32] - y_neg[32] - y_chg[32] + y_dis[32]
   + y_PV[32] - y_load[32] - 10.04300003051758 simga_cut_n[32,1]
   - 20.08600006103516 simga_cut_n[32,2]
   - 30.12900009155273 simga_cut_n[32,3]
   - 40.17200012207032 simga_cut_n[32,4]
   - 50.21500015258789 simga_cut_n[32,5]
   - 60.25800018310547 simga_cut_n[32,6]
   - 70.30100021362304 simga_cut_n[32,7]
   - 80.34400024414063 simga_cut_n[32,8]
   - 90.38700027465821 simga_cut_n[32,9] = -206.4677886962891
 c_power_balance_eq[33]: y_pos[33] - y_neg[33] - y_chg[33] + y_dis[33]
   + y_PV[33] - y_load[33] - 13.48500061035156 simga_cut_n[33,1]
   - 26.97000122070313 simga_cut_n[33,2]
   - 40.45500183105469 simga_cut_n[33,3]
   - 53.94000244140625 simga_cut_n[33,4]
   - 67.42500305175781 simga_cut_n[33,5]
   - 80.91000366210938 simga_cut_n[33,6]
   - 94.39500427246094 simga_cut_n[33,7]
   - 107.8800048828125 simga_cut_n[33,8]
   - 121.3650054931641 simga_cut_n[33,9] = -176.8464508056641
 c_power_balance_eq[34]: y_pos[34] - y_neg[34] - y_chg[34] + y_dis[34]
   + y_PV[34] - y_load[34] - 15.47200012207031 simga_cut_n[34,1]
   - 30.94400024414063 simga_cut_n[34,2]
   - 46.41600036621094 simga_cut_n[34,3]
   - 61.88800048828125 simga_cut_n[34,4]
   - 77.36000061035156 simga_cut_n[34,5]
   - 92.83200073242188 simga_cut_n[34,6]
   - 108.3040008544922 simga_cut_n[34,7]
   - 123.7760009765625 simga_cut_n[34,8]
   - 139.2480010986328 simga_cut_n[34,9] = -156.0933990478516
 c_power_balance_eq[35]: y_pos[35] - y_neg[35] - y_chg[35] + y_dis[35]
   + y_PV[35] - y_load[35] - 16.30223693847656 simga_cut_n[35,1]
   - 32.60447387695312 simga_cut_n[35,2]
   - 48.90671081542969 simga_cut_n[35,3]
   - 65.20894775390624 simga_cut_n[35,4]
   - 81.51118469238281 simga_cut_n[35,5]
   - 97.81342163085938 simga_cut_n[35,6]
   - 114.1156585693359 simga_cut_n[35,7]
   - 130.4178955078125 simga_cut_n[35,8]
   - 146.7201324462891 simga_cut_n[35,9]
   + 1.227763366699219 simga_add_n[35,1]
   + 2.455526733398437 simga_add_n[35,2]
   + 3.683290100097656 simga_add_n[35,3]
   + 4.911053466796875 simga_add_n[35,4]
   + 6.138816833496094 simga_add_n[35,5]
   + 7.366580200195313 simga_add_n[35,6]
   + 8.594343566894532 simga_add_n[35,7]
   + 9.82210693359375 simga_add_n[35,8]
   + 11.04987030029297 simga_add_n[35,9] = -137.7223663330078
 c_power_balance_eq[36]: y_pos[36] - y_neg[36] - y_chg[36] + y_dis[36]
   + y_PV[36] - y_load[36] - 17.67989807128906 simga_cut_n[36,1]
   - 35.35979614257813 simga_cut_n[36,2]
   - 53.03969421386719 simga_cut_n[36,3]
   - 70.71959228515625 simga_cut_n[36,4]
   - 88.39949035644531 simga_cut_n[36,5]
   - 106.0793884277344 simga_cut_n[36,6]
   - 123.7592864990234 simga_cut_n[36,7]
   - 141.4391845703125 simga_cut_n[36,8]
   - 159.1190826416016 simga_cut_n[36,9]
   + 4.313101196289063 simga_add_n[36,1]
   + 8.626202392578126 simga_add_n[36,2]
   + 12.93930358886719 simga_add_n[36,3]
   + 17.25240478515625 simga_add_n[36,4]
   + 21.56550598144531 simga_add_n[36,5]
   + 25.87860717773437 simga_add_n[36,6]
   + 30.19170837402344 simga_add_n[36,7]
   + 34.5048095703125 simga_add_n[36,8]
   + 38.81791076660156 simga_add_n[36,9] = -106.8689880371094
 c_power_balance_eq[37]: y_pos[37] - y_neg[37] - y_chg[37] + y_dis[37]
   + y_PV[37] - y_load[37] - 18.15235443115234 simga_cut_n[37,1]
   - 36.30470886230469 simga_cut_n[37,2]
   - 54.45706329345703 simga_cut_n[37,3]
   - 72.60941772460937 simga_cut_n[37,4]
   - 90.76177215576172 simga_cut_n[37,5]
   - 108.9141265869141 simga_cut_n[37,6]
   - 127.0664810180664 simga_cut_n[37,7]
   - 145.2188354492187 simga_cut_n[37,8]
   - 163.3711898803711 simga_cut_n[37,9]
   + 7.312644958496094 simga_add_n[37,1]
   + 14.62528991699219 simga_add_n[37,2]
   + 21.93793487548828 simga_add_n[37,3]
   + 29.25057983398438 simga_add_n[37,4]
   + 36.56322479248047 simga_add_n[37,5]
   + 43.87586975097656 simga_add_n[37,6]
   + 51.18851470947266 simga_add_n[37,7]
   + 58.50115966796875 simga_add_n[37,8]
   + 65.81380462646484 simga_add_n[37,9] = -76.87355041503906
 c_power_balance_eq[38]: y_pos[38] - y_neg[38] - y_chg[38] + y_dis[38]
   + y_PV[38] - y_load[38] - 18.81719269193793 simga_cut_n[38,1]
   - 37.63438538387587 simga_cut_n[38,2]
   - 56.4515780758138 simga_cut_n[38,3]
   - 75.26877076775173 simga_cut_n[38,4]
   - 94.08596345968967 simga_cut_n[38,5]
   - 112.9031561516276 simga_cut_n[38,6]
   - 131.7203488435655 simga_cut_n[38,7]
   - 150.5375415355035 simga_cut_n[38,8]
   - 169.3547342274414 simga_cut_n[38,9]
   + 9.112806087358942 simga_add_n[38,1]
   + 18.22561217471788 simga_add_n[38,2]
   + 27.33841826207682 simga_add_n[38,3]
   + 36.45122434943577 simga_add_n[38,4]
   + 45.56403043679471 simga_add_n[38,5]
   + 54.67683652415364 simga_add_n[38,6]
   + 63.78964261151259 simga_add_n[38,7]
   + 72.90244869887154 simga_add_n[38,8]
   + 82.01525478623049 simga_add_n[38,9] = -53.8306884765625
 c_power_balance_eq[39]: y_pos[39] - y_neg[39] - y_chg[39] + y_dis[39]
   + y_PV[39] - y_load[39] - 17.90970977035714 simga_cut_n[39,1]
   - 35.81941954071429 simga_cut_n[39,2]
   - 53.72912931107142 simga_cut_n[39,3]
   - 71.63883908142857 simga_cut_n[39,4]
   - 89.54854885178571 simga_cut_n[39,5]
   - 107.4582586221428 simga_cut_n[39,6] - 125.3679683925 simga_cut_n[39,7]
   - 143.2776781628571 simga_cut_n[39,8]
   - 161.1873879332143 simga_cut_n[39,9]
   + 11.36428925308036 simga_add_n[39,1]
   + 22.72857850616072 simga_add_n[39,2]
   + 34.09286775924108 simga_add_n[39,3]
   + 45.45715701232144 simga_add_n[39,4]
   + 56.8214462654018 simga_add_n[39,5]
   + 68.18573551848216 simga_add_n[39,6]
   + 79.55002477156252 simga_add_n[39,7]
   + 90.91431402464288 simga_add_n[39,8]
   + 102.2786032777232 simga_add_n[39,9] = -40.71728515624999
 c_power_balance_eq[40]: y_pos[40] - y_neg[40] - y_chg[40] + y_dis[40]
   + y_PV[40] - y_load[40] - 21.68679973842232 simga_cut_n[40,1]
   - 43.37359947684463 simga_cut_n[40,2]
   - 65.06039921526694 simga_cut_n[40,3]
   - 86.74719895368926 simga_cut_n[40,4]
   - 108.4339986921116 simga_cut_n[40,5]
   - 130.1207984305339 simga_cut_n[40,6]
   - 151.8075981689562 simga_cut_n[40,7]
   - 173.4943979073785 simga_cut_n[40,8]
   - 195.1811976458008 simga_cut_n[40,9]
   + 12.36420123814019 simga_add_n[40,1]
   + 24.72840247628037 simga_add_n[40,2]
   + 37.09260371442056 simga_add_n[40,3]
   + 49.45680495256074 simga_add_n[40,4]
   + 61.82100619070094 simga_add_n[40,5]
   + 74.18520742884112 simga_add_n[40,6]
   + 86.54940866698131 simga_add_n[40,7]
   + 98.91360990512149 simga_add_n[40,8]
   + 111.2778111432617 simga_add_n[40,9] = -13.35168457031251
 c_power_balance_eq[41]: y_pos[41] - y_neg[41] - y_chg[41] + y_dis[41]
   + y_PV[41] - y_load[41] - 23.03771037501046 simga_cut_n[41,1]
   - 46.07542075002091 simga_cut_n[41,2]
   - 69.11313112503136 simga_cut_n[41,3]
   - 92.15084150004182 simga_cut_n[41,4]
   - 115.1885518750523 simga_cut_n[41,5]
   - 138.2262622500627 simga_cut_n[41,6]
   - 161.2639726250732 simga_cut_n[41,7]
   - 184.3016830000836 simga_cut_n[41,8]
   - 207.3393933750941 simga_cut_n[41,9]
   + 13.13428974705986 simga_add_n[41,1]
   + 26.26857949411972 simga_add_n[41,2]
   + 39.40286924117957 simga_add_n[41,3]
   + 52.53715898823943 simga_add_n[41,4]
   + 65.67144873529929 simga_add_n[41,5]
   + 78.80573848235915 simga_add_n[41,6]
   + 91.94002822941901 simga_add_n[41,7]
   + 105.0743179764789 simga_add_n[41,8]
   + 118.2086077235387 simga_add_n[41,9] = 4.283386230468764
 c_power_balance_eq[42]: y_pos[42] - y_neg[42] - y_chg[42] + y_dis[42]
   + y_PV[42] - y_load[42] - 16.47894415424698 simga_cut_n[42,1]
   - 32.95788830849396 simga_cut_n[42,2]
   - 49.43683246274095 simga_cut_n[42,3]
   - 65.91577661698793 simga_cut_n[42,4]
   - 82.39472077123492 simga_cut_n[42,5]
   - 98.8736649254819 simga_cut_n[42,6]
   - 115.3526090797289 simga_cut_n[42,7]
   - 131.8315532339759 simga_cut_n[42,8]
   - 148.3104973882228 simga_cut_n[42,9]
   + 13.25205560161239 simga_add_n[42,1]
   + 26.50411120322478 simga_add_n[42,2]
   + 39.75616680483718 simga_add_n[42,3]
   + 53.00822240644957 simga_add_n[42,4]
   + 66.26027800806196 simga_add_n[42,5]
   + 79.51233360967436 simga_add_n[42,6]
   + 92.76438921128674 simga_add_n[42,7]
   + 106.0164448128991 simga_add_n[42,8]
   + 119.2685004145115 simga_add_n[42,9] = -48.56982421875
 c_power_balance_eq[43]: y_pos[43] - y_neg[43] - y_chg[43] + y_dis[43]
   + y_PV[43] - y_load[43] - 17.4226204773704 simga_cut_n[43,1]
   - 34.8452409547408 simga_cut_n[43,2]
   - 52.2678614321112 simga_cut_n[43,3]
   - 69.6904819094816 simga_cut_n[43,4]
   - 87.11310238685201 simga_cut_n[43,5]
   - 104.5357228642224 simga_cut_n[43,6]
   - 121.9583433415928 simga_cut_n[43,7]
   - 139.3809638189632 simga_cut_n[43,8]
   - 156.8035842963336 simga_cut_n[43,9]
   + 17.76337805778585 simga_add_n[43,1]
   + 35.5267561155717 simga_add_n[43,2]
   + 53.29013417335754 simga_add_n[43,3]
   + 71.0535122311434 simga_add_n[43,4]
   + 88.81689028892924 simga_add_n[43,5]
   + 106.5802683467151 simga_add_n[43,6]
   + 124.3436464045009 simga_add_n[43,7]
   + 142.1070244622868 simga_add_n[43,8]
   + 159.8704025200726 simga_add_n[43,9] = 43.80441284179688
 c_power_balance_eq[44]: y_pos[44] - y_neg[44] - y_chg[44] + y_dis[44]
   + y_PV[44] - y_load[44] - 23.75216920991493 simga_cut_n[44,1]
   - 47.50433841982986 simga_cut_n[44,2]
   - 71.25650762974479 simga_cut_n[44,3]
   - 95.00867683965971 simga_cut_n[44,4]
   - 118.7608460495746 simga_cut_n[44,5]
   - 142.5130152594896 simga_cut_n[44,6]
   - 166.2651844694045 simga_cut_n[44,7]
   - 190.0173536793194 simga_cut_n[44,8]
   - 213.7695228892343 simga_cut_n[44,9]
   + 17.91483213285851 simga_add_n[44,1]
   + 35.82966426571702 simga_add_n[44,2]
   + 53.74449639857553 simga_add_n[44,3]
   + 71.65932853143404 simga_add_n[44,4]
   + 89.57416066429255 simga_add_n[44,5]
   + 107.4889927971511 simga_add_n[44,6]
   + 125.4038249300096 simga_add_n[44,7]
   + 143.3186570628681 simga_add_n[44,8]
   + 161.2334891957266 simga_add_n[44,9] = 50.23709106445313
 c_power_balance_eq[45]: y_pos[45] - y_neg[45] - y_chg[45] + y_dis[45]
   + y_PV[45] - y_load[45] - 24.49579679190596 simga_cut_n[45,1]
   - 48.99159358381192 simga_cut_n[45,2]
   - 73.48739037571788 simga_cut_n[45,3]
   - 97.98318716762384 simga_cut_n[45,4]
   - 122.4789839595298 simga_cut_n[45,5]
   - 146.9747807514358 simga_cut_n[45,6]
   - 171.4705775433417 simga_cut_n[45,7]
   - 195.9663743352477 simga_cut_n[45,8]
   - 220.4621711271536 simga_cut_n[45,9]
   + 18.47520235360185 simga_add_n[45,1]
   + 36.9504047072037 simga_add_n[45,2]
   + 55.42560706080555 simga_add_n[45,3]
   + 73.90080941440741 simga_add_n[45,4]
   + 92.37601176800926 simga_add_n[45,5]
   + 110.8512141216111 simga_add_n[45,6]
   + 129.326416475213 simga_add_n[45,7]
   + 147.8016188288148 simga_add_n[45,8]
   + 166.2768211824167 simga_add_n[45,9] = 62.67355346679688
 c_power_balance_eq[46]: y_pos[46] - y_neg[46] - y_chg[46] + y_dis[46]
   + y_PV[46] - y_load[46] - 28.38617568876003 simga_cut_n[46,1]
   - 56.77235137752007 simga_cut_n[46,2]
   - 85.1585270662801 simga_cut_n[46,3]
   - 113.5447027550401 simga_cut_n[46,4]
   - 141.9308784438002 simga_cut_n[46,5]
   - 170.3170541325602 simga_cut_n[46,6]
   - 198.7032298213202 simga_cut_n[46,7]
   - 227.0894055100803 simga_cut_n[46,8]
   - 255.4755811988403 simga_cut_n[46,9]
   + 15.84882492159153 simga_add_n[46,1]
   + 31.69764984318306 simga_add_n[46,2]
   + 47.54647476477459 simga_add_n[46,3]
   + 63.39529968636612 simga_add_n[46,4]
   + 79.24412460795764 simga_add_n[46,5]
   + 95.09294952954917 simga_add_n[46,6]
   + 110.9417744511407 simga_add_n[46,7]
   + 126.7905993727322 simga_add_n[46,8]
   + 142.6394242943238 simga_add_n[46,9] = 71.9309997558594
 c_power_balance_eq[47]: y_pos[47] - y_neg[47] - y_chg[47] + y_dis[47]
   + y_PV[47] - y_load[47] - 28.8166116363088 simga_cut_n[47,1]
   - 57.6332232726176 simga_cut_n[47,2]
   - 86.44983490892641 simga_cut_n[47,3]
   - 115.2664465452352 simga_cut_n[47,4]
   - 144.083058181544 simga_cut_n[47,5]
   - 172.8996698178528 simga_cut_n[47,6]
   - 201.7162814541616 simga_cut_n[47,7]
   - 230.5328930904704 simga_cut_n[47,8]
   - 259.3495047267792 simga_cut_n[47,9]
   + 16.08938811955057 simga_add_n[47,1]
   + 32.17877623910115 simga_add_n[47,2]
   + 48.26816435865172 simga_add_n[47,3]
   + 64.35755247820229 simga_add_n[47,4]
   + 80.44694059775287 simga_add_n[47,5]
   + 96.53632871730345 simga_add_n[47,6]
   + 112.625716836854 simga_add_n[47,7]
   + 128.7151049564046 simga_add_n[47,8]
   + 144.8044930759552 simga_add_n[47,9] = 79.7137451171875
 c_power_balance_eq[48]: y_pos[48] - y_neg[48] - y_chg[48] + y_dis[48]
   + y_PV[48] - y_load[48] - 30.3288536067829 simga_cut_n[48,1]
   - 60.6577072135658 simga_cut_n[48,2]
   - 90.9865608203487 simga_cut_n[48,3]
   - 121.3154144271316 simga_cut_n[48,4]
   - 151.6442680339145 simga_cut_n[48,5]
   - 181.9731216406974 simga_cut_n[48,6]
   - 212.3019752474803 simga_cut_n[48,7]
   - 242.6308288542632 simga_cut_n[48,8]
   - 272.9596824610461 simga_cut_n[48,9]
   + 15.81114578286554 simga_add_n[48,1]
   + 31.62229156573108 simga_add_n[48,2]
   + 47.43343734859661 simga_add_n[48,3]
   + 63.24458313146215 simga_add_n[48,4]
   + 79.05572891432769 simga_add_n[48,5]
   + 94.86687469719323 simga_add_n[48,6]
   + 110.6780204800588 simga_add_n[48,7]
   + 126.4891662629243 simga_add_n[48,8]
   + 142.3003120457898 simga_add_n[48,9] = 82.2835693359375
 c_power_balance_eq[49]: y_pos[49] - y_neg[49] - y_chg[49] + y_dis[49]
   + y_PV[49] - y_load[49] - 32.12749037692363 simga_cut_n[49,1]
   - 64.25498075384726 simga_cut_n[49,2]
   - 96.38247113077091 simga_cut_n[49,3]
   - 128.5099615076945 simga_cut_n[49,4]
   - 160.6374518846182 simga_cut_n[49,5]
   - 192.7649422615418 simga_cut_n[49,6]
   - 224.8924326384654 simga_cut_n[49,7]
   - 257.0199230153891 simga_cut_n[49,8]
   - 289.1474133923127 simga_cut_n[49,9]
   + 16.74851059963887 simga_add_n[49,1]
   + 33.49702119927773 simga_add_n[49,2]
   + 50.24553179891659 simga_add_n[49,3]
   + 66.99404239855546 simga_add_n[49,4]
   + 83.74255299819433 simga_add_n[49,5]
   + 100.4910635978332 simga_add_n[49,6]
   + 117.2395741974721 simga_add_n[49,7]
   + 133.9880847971109 simga_add_n[49,8]
   + 150.7365953967498 simga_add_n[49,9] = 104.9044189453125
 c_power_balance_eq[50]: y_pos[50] - y_neg[50] - y_chg[50] + y_dis[50]
   + y_PV[50] - y_load[50] - 30.46193185373683 simga_cut_n[50,1]
   - 60.92386370747365 simga_cut_n[50,2]
   - 91.38579556121047 simga_cut_n[50,3]
   - 121.8477274149473 simga_cut_n[50,4]
   - 152.3096592686841 simga_cut_n[50,5]
   - 182.7715911224209 simga_cut_n[50,6]
   - 213.2335229761578 simga_cut_n[50,7]
   - 243.6954548298946 simga_cut_n[50,8]
   - 274.1573866836314 simga_cut_n[50,9]
   + 15.88006948903661 simga_add_n[50,1]
   + 31.76013897807322 simga_add_n[50,2]
   + 47.64020846710984 simga_add_n[50,3]
   + 63.52027795614644 simga_add_n[50,4]
   + 79.40034744518306 simga_add_n[50,5]
   + 95.28041693421967 simga_add_n[50,6]
   + 111.1604864232563 simga_add_n[50,7]
   + 127.0405559122929 simga_add_n[50,8]
   + 142.9206254013295 simga_add_n[50,9] = 85.12847900390625
 c_power_balance_eq[51]: y_pos[51] - y_neg[51] - y_chg[51] + y_dis[51]
   + y_PV[51] - y_load[51] - 30.91058977901737 simga_cut_n[51,1]
   - 61.82117955803474 simga_cut_n[51,2]
   - 92.7317693370521 simga_cut_n[51,3]
   - 123.6423591160695 simga_cut_n[51,4]
   - 154.5529488950868 simga_cut_n[51,5]
   - 185.4635386741042 simga_cut_n[51,6]
   - 216.3741284531216 simga_cut_n[51,7]
   - 247.2847182321389 simga_cut_n[51,8]
   - 278.1953080111563 simga_cut_n[51,9]
   + 16.11441022098263 simga_add_n[51,1]
   + 32.22882044196527 simga_add_n[51,2]
   + 48.3432306629479 simga_add_n[51,3]
   + 64.45764088393054 simga_add_n[51,4]
   + 80.57205110491317 simga_add_n[51,5]
   + 96.68646132589581 simga_add_n[51,6]
   + 112.8008715468784 simga_add_n[51,7]
   + 128.9152817678611 simga_add_n[51,8]
   + 145.0296919888437 simga_add_n[51,9] = 88.49118041992188
 c_power_balance_eq[52]: y_pos[52] - y_neg[52] - y_chg[52] + y_dis[52]
   + y_PV[52] - y_load[52] - 31.14318937414892 simga_cut_n[52,1]
   - 62.28637874829784 simga_cut_n[52,2]
   - 93.42956812244677 simga_cut_n[52,3]
   - 124.5727574965957 simga_cut_n[52,4]
   - 155.7159468707446 simga_cut_n[52,5]
   - 186.8591362448935 simga_cut_n[52,6]
   - 218.0023256190424 simga_cut_n[52,7]
   - 249.1455149931914 simga_cut_n[52,8]
   - 280.2887043673403 simga_cut_n[52,9]
   + 15.76581025964014 simga_add_n[52,1]
   + 31.53162051928028 simga_add_n[52,2]
   + 47.29743077892043 simga_add_n[52,3]
   + 63.06324103856057 simga_add_n[52,4]
   + 78.82905129820071 simga_add_n[52,5]
   + 94.59486155784086 simga_add_n[52,6]
   + 110.360671817481 simga_add_n[52,7]
   + 126.1264820771211 simga_add_n[52,8]
   + 141.8922923367613 simga_add_n[52,9] = 86.01339721679688
 c_power_balance_eq[53]: y_pos[53] - y_neg[53] - y_chg[53] + y_dis[53]
   + y_PV[53] - y_load[53] - 30.84708162501029 simga_cut_n[53,1]
   - 61.69416325002057 simga_cut_n[53,2]
   - 92.54124487503086 simga_cut_n[53,3]
   - 123.3883265000411 simga_cut_n[53,4]
   - 154.2354081250514 simga_cut_n[53,5]
   - 185.0824897500617 simga_cut_n[53,6]
   - 215.929571375072 simga_cut_n[53,7]
   - 246.7766530000823 simga_cut_n[53,8]
   - 277.6237346250926 simga_cut_n[53,9]
   + 15.61591886327097 simga_add_n[53,1]
   + 31.23183772654193 simga_add_n[53,2]
   + 46.8477565898129 simga_add_n[53,3]
   + 62.46367545308387 simga_add_n[53,4]
   + 78.07959431635483 simga_add_n[53,5]
   + 93.69551317962581 simga_add_n[53,6]
   + 109.3114320428968 simga_add_n[53,7]
   + 124.9273509061677 simga_add_n[53,8]
   + 140.5432697694387 simga_add_n[53,9] = 81.6545715332031
 c_power_balance_eq[54]: y_pos[54] - y_neg[54] - y_chg[54] + y_dis[54]
   + y_PV[54] - y_load[54] - 31.43929834392985 simga_cut_n[54,1]
   - 62.87859668785969 simga_cut_n[54,2]
   - 94.31789503178955 simga_cut_n[54,3]
   - 125.7571933757194 simga_cut_n[54,4]
   - 157.1964917196492 simga_cut_n[54,5]
   - 188.6357900635791 simga_cut_n[54,6]
   - 220.0750884075089 simga_cut_n[54,7]
   - 251.5143867514388 simga_cut_n[54,8]
   - 282.9536850953686 simga_cut_n[54,9]
   + 15.91570043536703 simga_add_n[54,1]
   + 31.83140087073406 simga_add_n[54,2]
   + 47.74710130610109 simga_add_n[54,3]
   + 63.66280174146812 simga_add_n[54,4]
   + 79.57850217683514 simga_add_n[54,5]
   + 95.49420261220217 simga_add_n[54,6]
   + 111.4099030475692 simga_add_n[54,7]
   + 127.3256034829362 simga_add_n[54,8]
   + 143.2413039183033 simga_add_n[54,9] = 88.92218017578125
 c_power_balance_eq[55]: y_pos[55] - y_neg[55] - y_chg[55] + y_dis[55]
   + y_PV[55] - y_load[55] - 29.82442895942964 simga_cut_n[55,1]
   - 59.64885791885928 simga_cut_n[55,2]
   - 89.47328687828892 simga_cut_n[55,3]
   - 119.2977158377186 simga_cut_n[55,4]
   - 149.1221447971482 simga_cut_n[55,5]
   - 178.9465737565778 simga_cut_n[55,6]
   - 208.7710027160075 simga_cut_n[55,7]
   - 238.5954316754371 simga_cut_n[55,8]
   - 268.4198606348667 simga_cut_n[55,9]
   + 16.1925723833438 simga_add_n[55,1]
   + 32.3851447666876 simga_add_n[55,2]
   + 48.5777171500314 simga_add_n[55,3]
   + 64.77028953337519 simga_add_n[55,4]
   + 80.96286191671899 simga_add_n[55,5]
   + 97.15543430006279 simga_add_n[55,6]
   + 113.3480066834066 simga_add_n[55,7]
   + 129.5405790667504 simga_add_n[55,8]
   + 145.7331514500942 simga_add_n[55,9] = 78.79574584960938
 c_power_balance_eq[56]: y_pos[56] - y_neg[56] - y_chg[56] + y_dis[56]
   + y_PV[56] - y_load[56] - 25.28332839737743 simga_cut_n[56,1]
   - 50.56665679475486 simga_cut_n[56,2]
   - 75.84998519213229 simga_cut_n[56,3]
   - 101.1333135895097 simga_cut_n[56,4]
   - 126.4166419868872 simga_cut_n[56,5]
   - 151.6999703842646 simga_cut_n[56,6]
   - 176.983298781642 simga_cut_n[56,7]
   - 202.2666271790195 simga_cut_n[56,8]
   - 227.5499555763969 simga_cut_n[56,9]
   + 17.03567184676319 simga_add_n[56,1]
   + 34.07134369352639 simga_add_n[56,2]
   + 51.10701554028958 simga_add_n[56,3]
   + 68.14268738705277 simga_add_n[56,4]
   + 85.17835923381597 simga_add_n[56,5]
   + 102.2140310805792 simga_add_n[56,6]
   + 119.2497029273424 simga_add_n[56,7]
   + 136.2853747741055 simga_add_n[56,8]
   + 153.3210466208687 simga_add_n[56,9] = 47.12933349609375
 c_power_balance_eq[57]: y_pos[57] - y_neg[57] - y_chg[57] + y_dis[57]
   + y_PV[57] - y_load[57] - 25.98666045995219 simga_cut_n[57,1]
   - 51.97332091990438 simga_cut_n[57,2]
   - 77.95998137985657 simga_cut_n[57,3]
   - 103.9466418398088 simga_cut_n[57,4]
   - 129.933302299761 simga_cut_n[57,5]
   - 155.9199627597131 simga_cut_n[57,6]
   - 181.9066232196653 simga_cut_n[57,7]
   - 207.8932836796175 simga_cut_n[57,8]
   - 233.8799441395697 simga_cut_n[57,9]
   + 17.50933868555562 simga_add_n[57,1]
   + 35.01867737111125 simga_add_n[57,2]
   + 52.52801605666686 simga_add_n[57,3]
   + 70.03735474222249 simga_add_n[57,4]
   + 87.54669342777811 simga_add_n[57,5]
   + 105.0560321133337 simga_add_n[57,6]
   + 122.5653707988893 simga_add_n[57,7]
   + 140.074709484445 simga_add_n[57,8]
   + 157.5840481700006 simga_add_n[57,9] = 54.64984130859375
 c_power_balance_eq[58]: y_pos[58] - y_neg[58] - y_chg[58] + y_dis[58]
   + y_PV[58] - y_load[58] - 25.54966036925494 simga_cut_n[58,1]
   - 51.09932073850989 simga_cut_n[58,2]
   - 76.64898110776483 simga_cut_n[58,3]
   - 102.1986414770198 simga_cut_n[58,4]
   - 127.7483018462747 simga_cut_n[58,5]
   - 153.2979622155297 simga_cut_n[58,6]
   - 178.8476225847846 simga_cut_n[58,7]
   - 204.3972829540396 simga_cut_n[58,8]
   - 229.9469433232945 simga_cut_n[58,9]
   + 17.21533902039349 simga_add_n[58,1]
   + 34.43067804078699 simga_add_n[58,2]
   + 51.64601706118049 simga_add_n[58,3]
   + 68.86135608157397 simga_add_n[58,4]
   + 86.07669510196747 simga_add_n[58,5]
   + 103.292034122361 simga_add_n[58,6]
   + 120.5073731427545 simga_add_n[58,7]
   + 137.7227121631479 simga_add_n[58,8]
   + 154.9380511835415 simga_add_n[58,9] = 50.51815795898438
 c_power_balance_eq[59]: y_pos[59] - y_neg[59] - y_chg[59] + y_dis[59]
   + y_PV[59] - y_load[59] - 27.44065962706981 simga_cut_n[59,1]
   - 54.88131925413961 simga_cut_n[59,2]
   - 82.32197888120942 simga_cut_n[59,3]
   - 109.7626385082792 simga_cut_n[59,4]
   - 137.203298135349 simga_cut_n[59,5]
   - 164.6439577624188 simga_cut_n[59,6]
   - 192.0846173894886 simga_cut_n[59,7]
   - 219.5252770165584 simga_cut_n[59,8]
   - 246.9659366436283 simga_cut_n[59,9]
   + 18.48933915222707 simga_add_n[59,1]
   + 36.97867830445414 simga_add_n[59,2]
   + 55.46801745668121 simga_add_n[59,3]
   + 73.95735660890827 simga_add_n[59,4]
   + 92.44669576113535 simga_add_n[59,5]
   + 110.9360349133624 simga_add_n[59,6]
   + 129.4253740655895 simga_add_n[59,7]
   + 147.9147132178165 simga_add_n[59,8]
   + 166.4040523700436 simga_add_n[59,9] = 74.4188232421875
 c_power_balance_eq[60]: y_pos[60] - y_neg[60] - y_chg[60] + y_dis[60]
   + y_PV[60] - y_load[60] - 24.92622129187907 simga_cut_n[60,1]
   - 49.85244258375813 simga_cut_n[60,2]
   - 74.77866387563719 simga_cut_n[60,3]
   - 99.70488516751627 simga_cut_n[60,4]
   - 124.6311064593953 simga_cut_n[60,5]
   - 149.5573277512744 simga_cut_n[60,6]
   - 174.4835490431535 simga_cut_n[60,7]
   - 199.4097703350325 simga_cut_n[60,8]
   - 224.3359916269116 simga_cut_n[60,9]
   + 17.16977785362874 simga_add_n[60,1]
   + 34.33955570725749 simga_add_n[60,2]
   + 51.50933356088624 simga_add_n[60,3]
   + 68.67911141451498 simga_add_n[60,4]
   + 85.84888926814372 simga_add_n[60,5]
   + 103.0186671217725 simga_add_n[60,6]
   + 120.1884449754012 simga_add_n[60,7]
   + 137.35822282903 simga_add_n[60,8]
   + 154.5280006826587 simga_add_n[60,9] = 48.08932495117188
 c_power_balance_eq[61]: y_pos[61] - y_neg[61] - y_chg[61] + y_dis[61]
   + y_PV[61] - y_load[61] - 23.20547884714184 simga_cut_n[61,1]
   - 46.41095769428368 simga_cut_n[61,2]
   - 69.61643654142551 simga_cut_n[61,3]
   - 92.82191538856736 simga_cut_n[61,4]
   - 116.0273942357092 simga_cut_n[61,5]
   - 139.232873082851 simga_cut_n[61,6]
   - 162.4383519299929 simga_cut_n[61,7]
   - 185.6438307771347 simga_cut_n[61,8]
   - 208.8493096242766 simga_cut_n[61,9]
   + 15.9845205425066 simga_add_n[61,1]
   + 31.9690410850132 simga_add_n[61,2]
   + 47.9535616275198 simga_add_n[61,3]
   + 63.93808217002639 simga_add_n[61,4]
   + 79.92260271253299 simga_add_n[61,5]
   + 95.90712325503959 simga_add_n[61,6]
   + 111.8916437975462 simga_add_n[61,7]
   + 127.8761643400528 simga_add_n[61,8]
   + 143.8606848825594 simga_add_n[61,9] = 21.52255249023438
 c_power_balance_eq[62]: y_pos[62] - y_neg[62] - y_chg[62] + y_dis[62]
   + y_PV[62] - y_load[62] - 18.55271517812684 simga_cut_n[62,1]
   - 37.10543035625368 simga_cut_n[62,2]
   - 55.65814553438052 simga_cut_n[62,3]
   - 74.21086071250735 simga_cut_n[62,4]
   - 92.7635758906342 simga_cut_n[62,5]
   - 111.316291068761 simga_cut_n[62,6]
   - 129.8690062468879 simga_cut_n[62,7]
   - 148.4217214250147 simga_cut_n[62,8]
   - 166.9744366031416 simga_cut_n[62,9]
   + 16.84328396738097 simga_add_n[62,1]
   + 33.68656793476195 simga_add_n[62,2]
   + 50.52985190214292 simga_add_n[62,3]
   + 67.37313586952389 simga_add_n[62,4]
   + 84.21641983690486 simga_add_n[62,5]
   + 101.0597038042858 simga_add_n[62,6]
   + 117.9029877716668 simga_add_n[62,7]
   + 134.7462717390478 simga_add_n[62,8]
   + 151.5895557064287 simga_add_n[62,9] = -8.2818603515625
 c_power_balance_eq[63]: y_pos[63] - y_neg[63] - y_chg[63] + y_dis[63]
   + y_PV[63] - y_load[63] - 25.11913585499702 simga_cut_n[63,1]
   - 50.23827170999405 simga_cut_n[63,2]
   - 75.35740756499106 simga_cut_n[63,3]
   - 100.4765434199881 simga_cut_n[63,4]
   - 125.5956792749851 simga_cut_n[63,5]
   - 150.7148151299821 simga_cut_n[63,6]
   - 175.8339509849792 simga_cut_n[63,7]
   - 200.9530868399762 simga_cut_n[63,8]
   - 226.0722226949732 simga_cut_n[63,9]
   + 14.71586475535454 simga_add_n[63,1]
   + 29.43172951070908 simga_add_n[63,2]
   + 44.14759426606362 simga_add_n[63,3]
   + 58.86345902141816 simga_add_n[63,4]
   + 73.5793237767727 simga_add_n[63,5]
   + 88.29518853212724 simga_add_n[63,6]
   + 103.0110532874818 simga_add_n[63,7]
   + 117.7269180428363 simga_add_n[63,8]
   + 132.4427827981909 simga_add_n[63,9] = 28.04626464843749
 c_power_balance_eq[64]: y_pos[64] - y_neg[64] - y_chg[64] + y_dis[64]
   + y_PV[64] - y_load[64] - 19.16049325766229 simga_cut_n[64,1]
   - 38.32098651532458 simga_cut_n[64,2]
   - 57.48147977298688 simga_cut_n[64,3]
   - 76.64197303064917 simga_cut_n[64,4]
   - 95.80246628831146 simga_cut_n[64,5]
   - 114.9629595459738 simga_cut_n[64,6]
   - 134.123452803636 simga_cut_n[64,7]
   - 153.2839460612983 simga_cut_n[64,8]
   - 172.4444393189606 simga_cut_n[64,9]
   + 10.83350698647833 simga_add_n[64,1]
   + 21.66701397295666 simga_add_n[64,2]
   + 32.50052095943499 simga_add_n[64,3]
   + 43.33402794591333 simga_add_n[64,4]
   + 54.16753493239166 simga_add_n[64,5]
   + 65.00104191886999 simga_add_n[64,6]
   + 75.83454890534833 simga_add_n[64,7]
   + 86.66805589182665 simga_add_n[64,8]
   + 97.50156287830499 simga_add_n[64,9] = -47.98492431640625
 c_power_balance_eq[65]: y_pos[65] - y_neg[65] - y_chg[65] + y_dis[65]
   + y_PV[65] - y_load[65] - 16.15287235167553 simga_cut_n[65,1]
   - 32.30574470335107 simga_cut_n[65,2]
   - 48.45861705502661 simga_cut_n[65,3]
   - 64.61148940670213 simga_cut_n[65,4]
   - 80.76436175837767 simga_cut_n[65,5]
   - 96.91723411005322 simga_cut_n[65,6]
   - 113.0701064617288 simga_cut_n[65,7]
   - 129.2229788134043 simga_cut_n[65,8]
   - 145.3758511650798 simga_cut_n[65,9]
   + 8.898127099008061 simga_add_n[65,1]
   + 17.79625419801612 simga_add_n[65,2]
   + 26.69438129702418 simga_add_n[65,3]
   + 35.59250839603224 simga_add_n[65,4]
   + 44.4906354950403 simga_add_n[65,5]
   + 53.38876259404837 simga_add_n[65,6]
   + 62.28688969305642 simga_add_n[65,7]
   + 71.18501679206449 simga_add_n[65,8]
   + 80.08314389107254 simga_add_n[65,9] = -53.71153259277342
 c_power_balance_eq[66]: y_pos[66] - y_neg[66] - y_chg[66] + y_dis[66]
   + y_PV[66] - y_load[66] - 13.18351872664955 simga_cut_n[66,1]
   - 26.3670374532991 simga_cut_n[66,2]
   - 39.55055617994865 simga_cut_n[66,3]
   - 52.7340749065982 simga_cut_n[66,4]
   - 65.91759363324775 simga_cut_n[66,5]
   - 79.1011123598973 simga_cut_n[66,6]
   - 92.28463108654685 simga_cut_n[66,7]
   - 105.4681498131964 simga_cut_n[66,8]
   - 118.6516685398459 simga_cut_n[66,9]
   + 1.026481883702014 simga_add_n[66,1]
   + 2.052963767404029 simga_add_n[66,2]
   + 3.079445651106043 simga_add_n[66,3]
   + 4.105927534808058 simga_add_n[66,4]
   + 5.132409418510072 simga_add_n[66,5]
   + 6.158891302212086 simga_add_n[66,6]
   + 7.1853731859141 simga_add_n[66,7]
   + 8.211855069616115 simga_add_n[66,8]
   + 9.238336953318129 simga_add_n[66,9] = -180.6471099853516
 c_power_balance_eq[67]: y_pos[67] - y_neg[67] - y_chg[67] + y_dis[67]
   + y_PV[67] - y_load[67] - 23.5277914127786 simga_cut_n[67,1]
   - 47.05558282555721 simga_cut_n[67,2]
   - 70.58337423833581 simga_cut_n[67,3]
   - 94.11116565111442 simga_cut_n[67,4]
   - 117.638957063893 simga_cut_n[67,5]
   - 141.1667484766716 simga_cut_n[67,6]
   - 164.6945398894502 simga_cut_n[67,7]
   - 188.2223313022288 simga_cut_n[67,8]
   - 211.7501227150074 simga_cut_n[67,9]
   + 0.6802087703268661 simga_add_n[67,1]
   + 1.360417540653732 simga_add_n[67,2]
   + 2.040626310980598 simga_add_n[67,3]
   + 2.720835081307464 simga_add_n[67,4]
   + 3.401043851634331 simga_add_n[67,5]
   + 4.081252621961196 simga_add_n[67,6]
   + 4.761461392288063 simga_add_n[67,7]
   + 5.441670162614929 simga_add_n[67,8]
   + 6.121878932941796 simga_add_n[67,9] = -95.89541625976563
 c_power_balance_eq[68]: y_pos[68] - y_neg[68] - y_chg[68] + y_dis[68]
   + y_PV[68] - y_load[68] - 10.69800033569336 simga_cut_n[68,1]
   - 21.39600067138672 simga_cut_n[68,2]
   - 32.09400100708008 simga_cut_n[68,3]
   - 42.79200134277344 simga_cut_n[68,4]
   - 53.4900016784668 simga_cut_n[68,5]
   - 64.18800201416016 simga_cut_n[68,6]
   - 74.88600234985351 simga_cut_n[68,7]
   - 85.58400268554688 simga_cut_n[68,8]
   - 96.28200302124023 simga_cut_n[68,9] = -207.3777923583984
 c_power_balance_eq[69]: y_pos[69] - y_neg[69] - y_chg[69] + y_dis[69]
   + y_PV[69] - y_load[69] - 11.94499969482422 simga_cut_n[69,1]
   - 23.88999938964844 simga_cut_n[69,2]
   - 35.83499908447266 simga_cut_n[69,3]
   - 47.77999877929688 simga_cut_n[69,4]
   - 59.72499847412109 simga_cut_n[69,5]
   - 71.66999816894531 simga_cut_n[69,6]
   - 83.61499786376953 simga_cut_n[69,7]
   - 95.55999755859375 simga_cut_n[69,8]
   - 107.504997253418 simga_cut_n[69,9] = -198.3513565063477
 c_power_balance_eq[70]: y_pos[70] - y_neg[70] - y_chg[70] + y_dis[70]
   + y_PV[70] - y_load[70] - 5.481000137329102 simga_cut_n[70,1]
   - 10.9620002746582 simga_cut_n[70,2]
   - 16.44300041198731 simga_cut_n[70,3]
   - 21.92400054931641 simga_cut_n[70,4]
   - 27.40500068664551 simga_cut_n[70,5]
   - 32.88600082397461 simga_cut_n[70,6]
   - 38.36700096130371 simga_cut_n[70,7]
   - 43.84800109863281 simga_cut_n[70,8]
   - 49.32900123596191 simga_cut_n[70,9] = -255.8374481201172
 c_power_balance_eq[71]: y_pos[71] - y_neg[71] - y_chg[71] + y_dis[71]
   + y_PV[71] - y_load[71] - 6.086000061035156 simga_cut_n[71,1]
   - 12.17200012207031 simga_cut_n[71,2]
   - 18.25800018310547 simga_cut_n[71,3]
   - 24.34400024414062 simga_cut_n[71,4]
   - 30.43000030517578 simga_cut_n[71,5]
   - 36.51600036621094 simga_cut_n[71,6]
   - 42.6020004272461 simga_cut_n[71,7]
   - 48.68800048828125 simga_cut_n[71,8]
   - 54.7740005493164 simga_cut_n[71,9] = -250.2237167358398
 c_power_balance_eq[72]: y_pos[72] - y_neg[72] - y_chg[72] + y_dis[72]
   + y_PV[72] - y_load[72] - 6.444999694824219 simga_cut_n[72,1]
   - 12.88999938964844 simga_cut_n[72,2]
   - 19.33499908447266 simga_cut_n[72,3]
   - 25.77999877929688 simga_cut_n[72,4]
   - 32.22499847412109 simga_cut_n[72,5]
   - 38.66999816894531 simga_cut_n[72,6]
   - 45.11499786376953 simga_cut_n[72,7]
   - 51.55999755859375 simga_cut_n[72,8]
   - 58.00499725341797 simga_cut_n[72,9] = -246.8179817199707
 c_power_balance_eq[73]: y_pos[73] - y_neg[73] - y_chg[73] + y_dis[73]
   + y_PV[73] - y_load[73] - 6.113999938964843 simga_cut_n[73,1]
   - 12.22799987792969 simga_cut_n[73,2]
   - 18.34199981689453 simga_cut_n[73,3]
   - 24.45599975585937 simga_cut_n[73,4]
   - 30.56999969482422 simga_cut_n[73,5]
   - 36.68399963378906 simga_cut_n[73,6]
   - 42.79799957275391 simga_cut_n[73,7]
   - 48.91199951171875 simga_cut_n[73,8]
   - 55.02599945068359 simga_cut_n[73,9] = -251.4588241577148
 c_power_balance_eq[74]: y_pos[74] - y_neg[74] - y_chg[74] + y_dis[74]
   + y_PV[74] - y_load[74] - 3.130999946594238 simga_cut_n[74,1]
   - 6.261999893188476 simga_cut_n[74,2]
   - 9.392999839782714 simga_cut_n[74,3]
   - 12.52399978637695 simga_cut_n[74,4]
   - 15.65499973297119 simga_cut_n[74,5]
   - 18.78599967956543 simga_cut_n[74,6]
   - 21.91699962615967 simga_cut_n[74,7]
   - 25.04799957275391 simga_cut_n[74,8]
   - 28.17899951934815 simga_cut_n[74,9] = -276.0864353179932
 c_power_balance_eq[75]: y_pos[75] - y_neg[75] - y_chg[75] + y_dis[75]
   + y_PV[75] - y_load[75] - 1.214999961853027 simga_cut_n[75,1]
   - 2.429999923706055 simga_cut_n[75,2]
   - 3.644999885559082 simga_cut_n[75,3]
   - 4.85999984741211 simga_cut_n[75,4]
   - 6.074999809265137 simga_cut_n[75,5]
   - 7.289999771118164 simga_cut_n[75,6]
   - 8.504999732971191 simga_cut_n[75,7]
   - 9.719999694824219 simga_cut_n[75,8]
   - 10.93499965667725 simga_cut_n[75,9] = -293.6135473251343
 c_power_balance_eq[76]: y_pos[76] - y_neg[76] - y_chg[76] + y_dis[76]
   + y_PV[76] - y_load[76] - 0.4710000038146973 simga_cut_n[76,1]
   - 0.9420000076293945 simga_cut_n[76,2]
   - 1.413000011444092 simga_cut_n[76,3]
   - 1.884000015258789 simga_cut_n[76,4]
   - 2.355000019073486 simga_cut_n[76,5]
   - 2.826000022888183 simga_cut_n[76,6]
   - 3.297000026702881 simga_cut_n[76,7]
   - 3.768000030517578 simga_cut_n[76,8]
   - 4.239000034332276 simga_cut_n[76,9] = -299.2932324409485
 c_power_balance_eq[77]: y_pos[77] - y_neg[77] - y_chg[77] + y_dis[77]
   + y_PV[77] - y_load[77] - 0.0360000014305115 simga_cut_n[77,1]
   - 0.072000002861023 simga_cut_n[77,2]
   - 0.1080000042915344 simga_cut_n[77,3]
   - 0.1440000057220459 simga_cut_n[77,4]
   - 0.1800000071525574 simga_cut_n[77,5]
   - 0.2160000085830688 simga_cut_n[77,6]
   - 0.2520000100135803 simga_cut_n[77,7]
   - 0.2880000114440918 simga_cut_n[77,8]
   - 0.3240000128746033 simga_cut_n[77,9] = -303.6910157203674
 c_power_balance_eq[78]: y_pos[78] - y_neg[78] - y_chg[78] + y_dis[78]
   + y_PV[78] - y_load[78] - 0.0360000014305115 simga_cut_n[78,1]
   - 0.072000002861023 simga_cut_n[78,2]
   - 0.1080000042915344 simga_cut_n[78,3]
   - 0.1440000057220459 simga_cut_n[78,4]
   - 0.1800000071525574 simga_cut_n[78,5]
   - 0.2160000085830688 simga_cut_n[78,6]
   - 0.2520000100135803 simga_cut_n[78,7]
   - 0.2880000114440918 simga_cut_n[78,8]
   - 0.3240000128746033 simga_cut_n[78,9] = -302.8810181617737
 c_power_balance_eq[79]: y_pos[79] - y_neg[79] - y_chg[79] + y_dis[79]
   + y_PV[79] - y_load[79] - 0.0360000014305115 simga_cut_n[79,1]
   - 0.072000002861023 simga_cut_n[79,2]
   - 0.1080000042915344 simga_cut_n[79,3]
   - 0.1440000057220459 simga_cut_n[79,4]
   - 0.1800000071525574 simga_cut_n[79,5]
   - 0.2160000085830688 simga_cut_n[79,6]
   - 0.2520000100135803 simga_cut_n[79,7]
   - 0.2880000114440918 simga_cut_n[79,8]
   - 0.3240000128746033 simga_cut_n[79,9] = -303.2610230445862
 c_power_balance_eq[80]: y_pos[80] - y_neg[80] - y_chg[80] + y_dis[80]
   + y_PV[80] - y_load[80] - 0.072000002861023 simga_cut_n[80,1]
   - 0.1440000057220459 simga_cut_n[80,2]
   - 0.2160000085830688 simga_cut_n[80,3]
   - 0.2880000114440918 simga_cut_n[80,4]
   - 0.3600000143051147 simga_cut_n[80,5]
   - 0.4320000171661377 simga_cut_n[80,6]
   - 0.5040000200271606 simga_cut_n[80,7]
   - 0.5760000228881836 simga_cut_n[80,8]
   - 0.6480000257492066 simga_cut_n[80,9] = -302.3020448684692
 c_power_balance_eq[81]: y_pos[81] - y_neg[81] - y_chg[81] + y_dis[81]
   + y_PV[81] - y_load[81] - 0.072000002861023 simga_cut_n[81,1]
   - 0.1440000057220459 simga_cut_n[81,2]
   - 0.2160000085830688 simga_cut_n[81,3]
   - 0.2880000114440918 simga_cut_n[81,4]
   - 0.3600000143051147 simga_cut_n[81,5]
   - 0.4320000171661377 simga_cut_n[81,6]
   - 0.5040000200271606 simga_cut_n[81,7]
   - 0.5760000228881836 simga_cut_n[81,8]
   - 0.6480000257492066 simga_cut_n[81,9] = -301.862042427063
 c_power_balance_eq[82]: y_pos[82] - y_neg[82] - y_chg[82] + y_dis[82]
   + y_PV[82] - y_load[82] - 0.0629999995231628 simga_cut_n[82,1]
   - 0.1259999990463257 simga_cut_n[82,2]
   - 0.1889999985694885 simga_cut_n[82,3]
   - 0.2519999980926514 simga_cut_n[82,4]
   - 0.3149999976158142 simga_cut_n[82,5]
   - 0.377999997138977 simga_cut_n[82,6]
   - 0.4409999966621399 simga_cut_n[82,7]
   - 0.5039999961853028 simga_cut_n[82,8]
   - 0.5669999957084656 simga_cut_n[82,9] = -302.9220399856567
 c_power_balance_eq[83]: y_pos[83] - y_neg[83] - y_chg[83] + y_dis[83]
   + y_PV[83] - y_load[83] - 0.0629999995231628 simga_cut_n[83,1]
   - 0.1259999990463257 simga_cut_n[83,2]
   - 0.1889999985694885 simga_cut_n[83,3]
   - 0.2519999980926514 simga_cut_n[83,4]
   - 0.3149999976158142 simga_cut_n[83,5]
   - 0.377999997138977 simga_cut_n[83,6]
   - 0.4409999966621399 simga_cut_n[83,7]
   - 0.5039999961853028 simga_cut_n[83,8]
   - 0.5669999957084656 simga_cut_n[83,9] = -301.5820436477661
 c_power_balance_eq[84]: y_pos[84] - y_neg[84] - y_chg[84] + y_dis[84]
   + y_PV[84] - y_load[84] - 0.1070000052452087 simga_cut_n[84,1]
   - 0.2140000104904175 simga_cut_n[84,2]
   - 0.3210000157356262 simga_cut_n[84,3]
   - 0.4280000209808349 simga_cut_n[84,4]
   - 0.5350000262260437 simga_cut_n[84,5]
   - 0.6420000314712524 simga_cut_n[84,6]
   - 0.7490000367164612 simga_cut_n[84,7]
   - 0.8560000419616699 simga_cut_n[84,8]
   - 0.9630000472068787 simga_cut_n[84,9] = -302.2030520439148
 c_power_balance_eq[85]: y_pos[85] - y_neg[85] - y_chg[85] + y_dis[85]
   + y_PV[85] - y_load[85] - 0.1070000052452087 simga_cut_n[85,1]
   - 0.2140000104904175 simga_cut_n[85,2]
   - 0.3210000157356262 simga_cut_n[85,3]
   - 0.4280000209808349 simga_cut_n[85,4]
   - 0.5350000262260437 simga_cut_n[85,5]
   - 0.6420000314712524 simga_cut_n[85,6]
   - 0.7490000367164612 simga_cut_n[85,7]
   - 0.8560000419616699 simga_cut_n[85,8]
   - 0.9630000472068787 simga_cut_n[85,9] = -301.7630496025085
 c_power_balance_eq[86]: y_pos[86] - y_neg[86] - y_chg[86] + y_dis[86]
   + y_PV[86] - y_load[86] - 0.1070000052452087 simga_cut_n[86,1]
   - 0.2140000104904175 simga_cut_n[86,2]
   - 0.3210000157356262 simga_cut_n[86,3]
   - 0.4280000209808349 simga_cut_n[86,4]
   - 0.5350000262260437 simga_cut_n[86,5]
   - 0.6420000314712524 simga_cut_n[86,6]
   - 0.7490000367164612 simga_cut_n[86,7]
   - 0.8560000419616699 simga_cut_n[86,8]
   - 0.9630000472068787 simga_cut_n[86,9] = -301.6230654716492
 c_power_balance_eq[87]: y_pos[87] - y_neg[87] - y_chg[87] + y_dis[87]
   + y_PV[87] - y_load[87] - 0.1070000052452087 simga_cut_n[87,1]
   - 0.2140000104904175 simga_cut_n[87,2]
   - 0.3210000157356262 simga_cut_n[87,3]
   - 0.4280000209808349 simga_cut_n[87,4]
   - 0.5350000262260437 simga_cut_n[87,5]
   - 0.6420000314712524 simga_cut_n[87,6]
   - 0.7490000367164612 simga_cut_n[87,7]
   - 0.8560000419616699 simga_cut_n[87,8]
   - 0.9630000472068787 simga_cut_n[87,9] = -302.5330386161804
 c_power_balance_eq[88]: y_pos[88] - y_neg[88] - y_chg[88] + y_dis[88]
   + y_PV[88] - y_load[88] - 0.1399999976158142 simga_cut_n[88,1]
   - 0.2799999952316284 simga_cut_n[88,2]
   - 0.4199999928474426 simga_cut_n[88,3]
   - 0.5599999904632569 simga_cut_n[88,4]
   - 0.699999988079071 simga_cut_n[88,5]
   - 0.8399999856948852 simga_cut_n[88,6]
   - 0.9799999833106995 simga_cut_n[88,7]
   - 1.119999980926514 simga_cut_n[88,8]
   - 1.259999978542328 simga_cut_n[88,9] = -301.4740543365479
 c_power_balance_eq[89]: y_pos[89] - y_neg[89] - y_chg[89] + y_dis[89]
   + y_PV[89] - y_load[89] - 0.1320000052452087 simga_cut_n[89,1]
   - 0.2640000104904175 simga_cut_n[89,2]
   - 0.3960000157356262 simga_cut_n[89,3]
   - 0.528000020980835 simga_cut_n[89,4]
   - 0.6600000262260437 simga_cut_n[89,5]
   - 0.7920000314712524 simga_cut_n[89,6]
   - 0.9240000367164611 simga_cut_n[89,7]
   - 1.05600004196167 simga_cut_n[89,8]
   - 1.188000047206879 simga_cut_n[89,9] = -302.3440799713135
 c_power_balance_eq[90]: y_pos[90] - y_neg[90] - y_chg[90] + y_dis[90]
   + y_PV[90] - y_load[90] - 0.1399999976158142 simga_cut_n[90,1]
   - 0.2799999952316284 simga_cut_n[90,2]
   - 0.4199999928474426 simga_cut_n[90,3]
   - 0.5599999904632569 simga_cut_n[90,4]
   - 0.699999988079071 simga_cut_n[90,5]
   - 0.8399999856948852 simga_cut_n[90,6]
   - 0.9799999833106995 simga_cut_n[90,7]
   - 1.119999980926514 simga_cut_n[90,8]
   - 1.259999978542328 simga_cut_n[90,9] = -302.6740665435791
 c_power_balance_eq[91]: y_pos[91] - y_neg[91] - y_chg[91] + y_dis[91]
   + y_PV[91] - y_load[91] - 0.1399999976158142 simga_cut_n[91,1]
   - 0.2799999952316284 simga_cut_n[91,2]
   - 0.4199999928474426 simga_cut_n[91,3]
   - 0.5599999904632569 simga_cut_n[91,4]
   - 0.699999988079071 simga_cut_n[91,5]
   - 0.8399999856948852 simga_cut_n[91,6]
   - 0.9799999833106995 simga_cut_n[91,7]
   - 1.119999980926514 simga_cut_n[91,8]
   - 1.259999978542328 simga_cut_n[91,9] = -302.1440677642822
 c_power_balance_eq[92]: y_pos[92] - y_neg[92] - y_chg[92] + y_dis[92]
   + y_PV[92] - y_load[92] - 0.172000002861023 simga_cut_n[92,1]
   - 0.3440000057220459 simga_cut_n[92,2]
   - 0.5160000085830688 simga_cut_n[92,3]
   - 0.6880000114440918 simga_cut_n[92,4]
   - 0.8600000143051147 simga_cut_n[92,5]
   - 1.032000017166138 simga_cut_n[92,6]
   - 1.204000020027161 simga_cut_n[92,7]
   - 1.376000022888184 simga_cut_n[92,8]
   - 1.548000025749207 simga_cut_n[92,9] = -303.7750859260559
 c_power_balance_eq[93]: y_pos[93] - y_neg[93] - y_chg[93] + y_dis[93]
   + y_PV[93] - y_load[93] - 0.1669999957084656 simga_cut_n[93,1]
   - 0.3339999914169312 simga_cut_n[93,2]
   - 0.5009999871253967 simga_cut_n[93,3]
   - 0.6679999828338623 simga_cut_n[93,4]
   - 0.8349999785423279 simga_cut_n[93,5]
   - 1.001999974250793 simga_cut_n[93,6]
   - 1.168999969959259 simga_cut_n[93,7]
   - 1.335999965667725 simga_cut_n[93,8]
   - 1.50299996137619 simga_cut_n[93,9] = -302.9650883674622
 c_power_balance_eq[94]: y_pos[94] - y_neg[94] - y_chg[94] + y_dis[94]
   + y_PV[94] - y_load[94] - 0.172000002861023 simga_cut_n[94,1]
   - 0.3440000057220459 simga_cut_n[94,2]
   - 0.5160000085830688 simga_cut_n[94,3]
   - 0.6880000114440918 simga_cut_n[94,4]
   - 0.8600000143051147 simga_cut_n[94,5]
   - 1.032000017166138 simga_cut_n[94,6]
   - 1.204000020027161 simga_cut_n[94,7]
   - 1.376000022888184 simga_cut_n[94,8]
   - 1.548000025749207 simga_cut_n[94,9] = -304.2050786018372
 c_power_balance_eq[95]: y_pos[95] - y_neg[95] - y_chg[95] + y_dis[95]
   + y_PV[95] - y_load[95] - 0.172000002861023 simga_cut_n[95,1]
   - 0.3440000057220459 simga_cut_n[95,2]
   - 0.5160000085830688 simga_cut_n[95,3]
   - 0.6880000114440918 simga_cut_n[95,4]
   - 0.8600000143051147 simga_cut_n[95,5]
   - 1.032000017166138 simga_cut_n[95,6]
   - 1.204000020027161 simga_cut_n[95,7]
   - 1.376000022888184 simga_cut_n[95,8]
   - 1.548000025749207 simga_cut_n[95,9] = -304.3050847053528
 c_y_cut_conv_combin[0]: simga_cut_n[0,0] + simga_cut_n[0,1]
   + simga_cut_n[0,2] + simga_cut_n[0,3] + simga_cut_n[0,4]
   + simga_cut_n[0,5] + simga_cut_n[0,6] + simga_cut_n[0,7]
   + simga_cut_n[0,8] + simga_cut_n[0,9] = 1
 c_y_cut_conv_combin[1]: simga_cut_n[1,0] + simga_cut_n[1,1]
   + simga_cut_n[1,2] + simga_cut_n[1,3] + simga_cut_n[1,4]
   + simga_cut_n[1,5] + simga_cut_n[1,6] + simga_cut_n[1,7]
   + simga_cut_n[1,8] + simga_cut_n[1,9] = 1
 c_y_cut_conv_combin[2]: simga_cut_n[2,0] + simga_cut_n[2,1]
   + simga_cut_n[2,2] + simga_cut_n[2,3] + simga_cut_n[2,4]
   + simga_cut_n[2,5] + simga_cut_n[2,6] + simga_cut_n[2,7]
   + simga_cut_n[2,8] + simga_cut_n[2,9] = 1
 c_y_cut_conv_combin[3]: simga_cut_n[3,0] + simga_cut_n[3,1]
   + simga_cut_n[3,2] + simga_cut_n[3,3] + simga_cut_n[3,4]
   + simga_cut_n[3,5] + simga_cut_n[3,6] + simga_cut_n[3,7]
   + simga_cut_n[3,8] + simga_cut_n[3,9] = 1
 c_y_cut_conv_combin[4]: simga_cut_n[4,0] + simga_cut_n[4,1]
   + simga_cut_n[4,2] + simga_cut_n[4,3] + simga_cut_n[4,4]
   + simga_cut_n[4,5] + simga_cut_n[4,6] + simga_cut_n[4,7]
   + simga_cut_n[4,8] + simga_cut_n[4,9] = 1
 c_y_cut_conv_combin[5]: simga_cut_n[5,0] + simga_cut_n[5,1]
   + simga_cut_n[5,2] + simga_cut_n[5,3] + simga_cut_n[5,4]
   + simga_cut_n[5,5] + simga_cut_n[5,6] + simga_cut_n[5,7]
   + simga_cut_n[5,8] + simga_cut_n[5,9] = 1
 c_y_cut_conv_combin[6]: simga_cut_n[6,0] + simga_cut_n[6,1]
   + simga_cut_n[6,2] + simga_cut_n[6,3] + simga_cut_n[6,4]
   + simga_cut_n[6,5] + simga_cut_n[6,6] + simga_cut_n[6,7]
   + simga_cut_n[6,8] + simga_cut_n[6,9] = 1
 c_y_cut_conv_combin[7]: simga_cut_n[7,0] + simga_cut_n[7,1]
   + simga_cut_n[7,2] + simga_cut_n[7,3] + simga_cut_n[7,4]
   + simga_cut_n[7,5] + simga_cut_n[7,6] + simga_cut_n[7,7]
   + simga_cut_n[7,8] + simga_cut_n[7,9] = 1
 c_y_cut_conv_combin[8]: simga_cut_n[8,0] + simga_cut_n[8,1]
   + simga_cut_n[8,2] + simga_cut_n[8,3] + simga_cut_n[8,4]
   + simga_cut_n[8,5] + simga_cut_n[8,6] + simga_cut_n[8,7]
   + simga_cut_n[8,8] + simga_cut_n[8,9] = 1
 c_y_cut_conv_combin[9]: simga_cut_n[9,0] + simga_cut_n[9,1]
   + simga_cut_n[9,2] + simga_cut_n[9,3] + simga_cut_n[9,4]
   + simga_cut_n[9,5] + simga_cut_n[9,6] + simga_cut_n[9,7]
   + simga_cut_n[9,8] + simga_cut_n[9,9] = 1
 c_y_cut_conv_combin[10]: simga_cut_n[10,0] + simga_cut_n[10,1]
   + simga_cut_n[10,2] + simga_cut_n[10,3] + simga_cut_n[10,4]
   + simga_cut_n[10,5] + simga_cut_n[10,6] + simga_cut_n[10,7]
   + simga_cut_n[10,8] + simga_cut_n[10,9] = 1
 c_y_cut_conv_combin[11]: simga_cut_n[11,0] + simga_cut_n[11,1]
   + simga_cut_n[11,2] + simga_cut_n[11,3] + simga_cut_n[11,4]
   + simga_cut_n[11,5] + simga_cut_n[11,6] + simga_cut_n[11,7]
   + simga_cut_n[11,8] + simga_cut_n[11,9] = 1
 c_y_cut_conv_combin[12]: simga_cut_n[12,0] + simga_cut_n[12,1]
   + simga_cut_n[12,2] + simga_cut_n[12,3] + simga_cut_n[12,4]
   + simga_cut_n[12,5] + simga_cut_n[12,6] + simga_cut_n[12,7]
   + simga_cut_n[12,8] + simga_cut_n[12,9] = 1
 c_y_cut_conv_combin[13]: simga_cut_n[13,0] + simga_cut_n[13,1]
   + simga_cut_n[13,2] + simga_cut_n[13,3] + simga_cut_n[13,4]
   + simga_cut_n[13,5] + simga_cut_n[13,6] + simga_cut_n[13,7]
   + simga_cut_n[13,8] + simga_cut_n[13,9] = 1
 c_y_cut_conv_combin[14]: simga_cut_n[14,0] + simga_cut_n[14,1]
   + simga_cut_n[14,2] + simga_cut_n[14,3] + simga_cut_n[14,4]
   + simga_cut_n[14,5] + simga_cut_n[14,6] + simga_cut_n[14,7]
   + simga_cut_n[14,8] + simga_cut_n[14,9] = 1
 c_y_cut_conv_combin[15]: simga_cut_n[15,0] + simga_cut_n[15,1]
   + simga_cut_n[15,2] + simga_cut_n[15,3] + simga_cut_n[15,4]
   + simga_cut_n[15,5] + simga_cut_n[15,6] + simga_cut_n[15,7]
   + simga_cut_n[15,8] + simga_cut_n[15,9] = 1
 c_y_cut_conv_combin[16]: simga_cut_n[16,0] + simga_cut_n[16,1]
   + simga_cut_n[16,2] + simga_cut_n[16,3] + simga_cut_n[16,4]
   + simga_cut_n[16,5] + simga_cut_n[16,6] + simga_cut_n[16,7]
   + simga_cut_n[16,8] + simga_cut_n[16,9] = 1
 c_y_cut_conv_combin[17]: simga_cut_n[17,0] + simga_cut_n[17,1]
   + simga_cut_n[17,2] + simga_cut_n[17,3] + simga_cut_n[17,4]
   + simga_cut_n[17,5] + simga_cut_n[17,6] + simga_cut_n[17,7]
   + simga_cut_n[17,8] + simga_cut_n[17,9] = 1
 c_y_cut_conv_combin[18]: simga_cut_n[18,0] + simga_cut_n[18,1]
   + simga_cut_n[18,2] + simga_cut_n[18,3] + simga_cut_n[18,4]
   + simga_cut_n[18,5] + simga_cut_n[18,6] + simga_cut_n[18,7]
   + simga_cut_n[18,8] + simga_cut_n[18,9] = 1
 c_y_cut_conv_combin[19]: simga_cut_n[19,0] + simga_cut_n[19,1]
   + simga_cut_n[19,2] + simga_cut_n[19,3] + simga_cut_n[19,4]
   + simga_cut_n[19,5] + simga_cut_n[19,6] + simga_cut_n[19,7]
   + simga_cut_n[19,8] + simga_cut_n[19,9] = 1
 c_y_cut_conv_combin[20]: simga_cut_n[20,0] + simga_cut_n[20,1]
   + simga_cut_n[20,2] + simga_cut_n[20,3] + simga_cut_n[20,4]
   + simga_cut_n[20,5] + simga_cut_n[20,6] + simga_cut_n[20,7]
   + simga_cut_n[20,8] + simga_cut_n[20,9] = 1
 c_y_cut_conv_combin[21]: simga_cut_n[21,0] + simga_cut_n[21,1]
   + simga_cut_n[21,2] + simga_cut_n[21,3] + simga_cut_n[21,4]
   + simga_cut_n[21,5] + simga_cut_n[21,6] + simga_cut_n[21,7]
   + simga_cut_n[21,8] + simga_cut_n[21,9] = 1
 c_y_cut_conv_combin[22]: simga_cut_n[22,0] + simga_cut_n[22,1]
   + simga_cut_n[22,2] + simga_cut_n[22,3] + simga_cut_n[22,4]
   + simga_cut_n[22,5] + simga_cut_n[22,6] + simga_cut_n[22,7]
   + simga_cut_n[22,8] + simga_cut_n[22,9] = 1
 c_y_cut_conv_combin[23]: simga_cut_n[23,0] + simga_cut_n[23,1]
   + simga_cut_n[23,2] + simga_cut_n[23,3] + simga_cut_n[23,4]
   + simga_cut_n[23,5] + simga_cut_n[23,6] + simga_cut_n[23,7]
   + simga_cut_n[23,8] + simga_cut_n[23,9] = 1
 c_y_cut_conv_combin[24]: simga_cut_n[24,0] + simga_cut_n[24,1]
   + simga_cut_n[24,2] + simga_cut_n[24,3] + simga_cut_n[24,4]
   + simga_cut_n[24,5] + simga_cut_n[24,6] + simga_cut_n[24,7]
   + simga_cut_n[24,8] + simga_cut_n[24,9] = 1
 c_y_cut_conv_combin[25]: simga_cut_n[25,0] + simga_cut_n[25,1]
   + simga_cut_n[25,2] + simga_cut_n[25,3] + simga_cut_n[25,4]
   + simga_cut_n[25,5] + simga_cut_n[25,6] + simga_cut_n[25,7]
   + simga_cut_n[25,8] + simga_cut_n[25,9] = 1
 c_y_cut_conv_combin[26]: simga_cut_n[26,0] + simga_cut_n[26,1]
   + simga_cut_n[26,2] + simga_cut_n[26,3] + simga_cut_n[26,4]
   + simga_cut_n[26,5] + simga_cut_n[26,6] + simga_cut_n[26,7]
   + simga_cut_n[26,8] + simga_cut_n[26,9] = 1
 c_y_cut_conv_combin[27]: simga_cut_n[27,0] + simga_cut_n[27,1]
   + simga_cut_n[27,2] + simga_cut_n[27,3] + simga_cut_n[27,4]
   + simga_cut_n[27,5] + simga_cut_n[27,6] + simga_cut_n[27,7]
   + simga_cut_n[27,8] + simga_cut_n[27,9] = 1
 c_y_cut_conv_combin[28]: simga_cut_n[28,0] + simga_cut_n[28,1]
   + simga_cut_n[28,2] + simga_cut_n[28,3] + simga_cut_n[28,4]
   + simga_cut_n[28,5] + simga_cut_n[28,6] + simga_cut_n[28,7]
   + simga_cut_n[28,8] + simga_cut_n[28,9] = 1
 c_y_cut_conv_combin[29]: simga_cut_n[29,0] + simga_cut_n[29,1]
   + simga_cut_n[29,2] + simga_cut_n[29,3] + simga_cut_n[29,4]
   + simga_cut_n[29,5] + simga_cut_n[29,6] + simga_cut_n[29,7]
   + simga_cut_n[29,8] + simga_cut_n[29,9] = 1
 c_y_cut_conv_combin[30]: simga_cut_n[30,0] + simga_cut_n[30,1]
   + simga_cut_n[30,2] + simga_cut_n[30,3] + simga_cut_n[30,4]
   + simga_cut_n[30,5] + simga_cut_n[30,6] + simga_cut_n[30,7]
   + simga_cut_n[30,8] + simga_cut_n[30,9] = 1
 c_y_cut_conv_combin[31]: simga_cut_n[31,0] + simga_cut_n[31,1]
   + simga_cut_n[31,2] + simga_cut_n[31,3] + simga_cut_n[31,4]
   + simga_cut_n[31,5] + simga_cut_n[31,6] + simga_cut_n[31,7]
   + simga_cut_n[31,8] + simga_cut_n[31,9] = 1
 c_y_cut_conv_combin[32]: simga_cut_n[32,0] + simga_cut_n[32,1]
   + simga_cut_n[32,2] + simga_cut_n[32,3] + simga_cut_n[32,4]
   + simga_cut_n[32,5] + simga_cut_n[32,6] + simga_cut_n[32,7]
   + simga_cut_n[32,8] + simga_cut_n[32,9] = 1
 c_y_cut_conv_combin[33]: simga_cut_n[33,0] + simga_cut_n[33,1]
   + simga_cut_n[33,2] + simga_cut_n[33,3] + simga_cut_n[33,4]
   + simga_cut_n[33,5] + simga_cut_n[33,6] + simga_cut_n[33,7]
   + simga_cut_n[33,8] + simga_cut_n[33,9] = 1
 c_y_cut_conv_combin[34]: simga_cut_n[34,0] + simga_cut_n[34,1]
   + simga_cut_n[34,2] + simga_cut_n[34,3] + simga_cut_n[34,4]
   + simga_cut_n[34,5] + simga_cut_n[34,6] + simga_cut_n[34,7]
   + simga_cut_n[34,8] + simga_cut_n[34,9] = 1
 c_y_cut_conv_combin[35]: simga_cut_n[35,0] + simga_cut_n[35,1]
   + simga_cut_n[35,2] + simga_cut_n[35,3] + simga_cut_n[35,4]
   + simga_cut_n[35,5] + simga_cut_n[35,6] + simga_cut_n[35,7]
   + simga_cut_n[35,8] + simga_cut_n[35,9] = 1
 c_y_cut_conv_combin[36]: simga_cut_n[36,0] + simga_cut_n[36,1]
   + simga_cut_n[36,2] + simga_cut_n[36,3] + simga_cut_n[36,4]
   + simga_cut_n[36,5] + simga_cut_n[36,6] + simga_cut_n[36,7]
   + simga_cut_n[36,8] + simga_cut_n[36,9] = 1
 c_y_cut_conv_combin[37]: simga_cut_n[37,0] + simga_cut_n[37,1]
   + simga_cut_n[37,2] + simga_cut_n[37,3] + simga_cut_n[37,4]
   + simga_cut_n[37,5] + simga_cut_n[37,6] + simga_cut_n[37,7]
   + simga_cut_n[37,8] + simga_cut_n[37,9] = 1
 c_y_cut_conv_combin[38]: simga_cut_n[38,0] + simga_cut_n[38,1]
   + simga_cut_n[38,2] + simga_cut_n[38,3] + simga_cut_n[38,4]
   + simga_cut_n[38,5] + simga_cut_n[38,6] + simga_cut_n[38,7]
   + simga_cut_n[38,8] + simga_cut_n[38,9] = 1
 c_y_cut_conv_combin[39]: simga_cut_n[39,0] + simga_cut_n[39,1]
   + simga_cut_n[39,2] + simga_cut_n[39,3] + simga_cut_n[39,4]
   + simga_cut_n[39,5] + simga_cut_n[39,6] + simga_cut_n[39,7]
   + simga_cut_n[39,8] + simga_cut_n[39,9] = 1
 c_y_cut_conv_combin[40]: simga_cut_n[40,0] + simga_cut_n[40,1]
   + simga_cut_n[40,2] + simga_cut_n[40,3] + simga_cut_n[40,4]
   + simga_cut_n[40,5] + simga_cut_n[40,6] + simga_cut_n[40,7]
   + simga_cut_n[40,8] + simga_cut_n[40,9] = 1
 c_y_cut_conv_combin[41]: simga_cut_n[41,0] + simga_cut_n[41,1]
   + simga_cut_n[41,2] + simga_cut_n[41,3] + simga_cut_n[41,4]
   + simga_cut_n[41,5] + simga_cut_n[41,6] + simga_cut_n[41,7]
   + simga_cut_n[41,8] + simga_cut_n[41,9] = 1
 c_y_cut_conv_combin[42]: simga_cut_n[42,0] + simga_cut_n[42,1]
   + simga_cut_n[42,2] + simga_cut_n[42,3] + simga_cut_n[42,4]
   + simga_cut_n[42,5] + simga_cut_n[42,6] + simga_cut_n[42,7]
   + simga_cut_n[42,8] + simga_cut_n[42,9] = 1
 c_y_cut_conv_combin[43]: simga_cut_n[43,0] + simga_cut_n[43,1]
   + simga_cut_n[43,2] + simga_cut_n[43,3] + simga_cut_n[43,4]
   + simga_cut_n[43,5] + simga_cut_n[43,6] + simga_cut_n[43,7]
   + simga_cut_n[43,8] + simga_cut_n[43,9] = 1
 c_y_cut_conv_combin[44]: simga_cut_n[44,0] + simga_cut_n[44,1]
   + simga_cut_n[44,2] + simga_cut_n[44,3] + simga_cut_n[44,4]
   + simga_cut_n[44,5] + simga_cut_n[44,6] + simga_cut_n[44,7]
   + simga_cut_n[44,8] + simga_cut_n[44,9] = 1
 c_y_cut_conv_combin[45]: simga_cut_n[45,0] + simga_cut_n[45,1]
   + simga_cut_n[45,2] + simga_cut_n[45,3] + simga_cut_n[45,4]
   + simga_cut_n[45,5] + simga_cut_n[45,6] + simga_cut_n[45,7]
   + simga_cut_n[45,8] + simga_cut_n[45,9] = 1
 c_y_cut_conv_combin[46]: simga_cut_n[46,0] + simga_cut_n[46,1]
   + simga_cut_n[46,2] + simga_cut_n[46,3] + simga_cut_n[46,4]
   + simga_cut_n[46,5] + simga_cut_n[46,6] + simga_cut_n[46,7]
   + simga_cut_n[46,8] + simga_cut_n[46,9] = 1
 c_y_cut_conv_combin[47]: simga_cut_n[47,0] + simga_cut_n[47,1]
   + simga_cut_n[47,2] + simga_cut_n[47,3] + simga_cut_n[47,4]
   + simga_cut_n[47,5] + simga_cut_n[47,6] + simga_cut_n[47,7]
   + simga_cut_n[47,8] + simga_cut_n[47,9] = 1
 c_y_cut_conv_combin[48]: simga_cut_n[48,0] + simga_cut_n[48,1]
   + simga_cut_n[48,2] + simga_cut_n[48,3] + simga_cut_n[48,4]
   + simga_cut_n[48,5] + simga_cut_n[48,6] + simga_cut_n[48,7]
   + simga_cut_n[48,8] + simga_cut_n[48,9] = 1
 c_y_cut_conv_combin[49]: simga_cut_n[49,0] + simga_cut_n[49,1]
   + simga_cut_n[49,2] + simga_cut_n[49,3] + simga_cut_n[49,4]
   + simga_cut_n[49,5] + simga_cut_n[49,6] + simga_cut_n[49,7]
   + simga_cut_n[49,8] + simga_cut_n[49,9] = 1
 c_y_cut_conv_combin[50]: simga_cut_n[50,0] + simga_cut_n[50,1]
   + simga_cut_n[50,2] + simga_cut_n[50,3] + simga_cut_n[50,4]
   + simga_cut_n[50,5] + simga_cut_n[50,6] + simga_cut_n[50,7]
   + simga_cut_n[50,8] + simga_cut_n[50,9] = 1
 c_y_cut_conv_combin[51]: simga_cut_n[51,0] + simga_cut_n[51,1]
   + simga_cut_n[51,2] + simga_cut_n[51,3] + simga_cut_n[51,4]
   + simga_cut_n[51,5] + simga_cut_n[51,6] + simga_cut_n[51,7]
   + simga_cut_n[51,8] + simga_cut_n[51,9] = 1
 c_y_cut_conv_combin[52]: simga_cut_n[52,0] + simga_cut_n[52,1]
   + simga_cut_n[52,2] + simga_cut_n[52,3] + simga_cut_n[52,4]
   + simga_cut_n[52,5] + simga_cut_n[52,6] + simga_cut_n[52,7]
   + simga_cut_n[52,8] + simga_cut_n[52,9] = 1
 c_y_cut_conv_combin[53]: simga_cut_n[53,0] + simga_cut_n[53,1]
   + simga_cut_n[53,2] + simga_cut_n[53,3] + simga_cut_n[53,4]
   + simga_cut_n[53,5] + simga_cut_n[53,6] + simga_cut_n[53,7]
   + simga_cut_n[53,8] + simga_cut_n[53,9] = 1
 c_y_cut_conv_combin[54]: simga_cut_n[54,0] + simga_cut_n[54,1]
   + simga_cut_n[54,2] + simga_cut_n[54,3] + simga_cut_n[54,4]
   + simga_cut_n[54,5] + simga_cut_n[54,6] + simga_cut_n[54,7]
   + simga_cut_n[54,8] + simga_cut_n[54,9] = 1
 c_y_cut_conv_combin[55]: simga_cut_n[55,0] + simga_cut_n[55,1]
   + simga_cut_n[55,2] + simga_cut_n[55,3] + simga_cut_n[55,4]
   + simga_cut_n[55,5] + simga_cut_n[55,6] + simga_cut_n[55,7]
   + simga_cut_n[55,8] + simga_cut_n[55,9] = 1
 c_y_cut_conv_combin[56]: simga_cut_n[56,0] + simga_cut_n[56,1]
   + simga_cut_n[56,2] + simga_cut_n[56,3] + simga_cut_n[56,4]
   + simga_cut_n[56,5] + simga_cut_n[56,6] + simga_cut_n[56,7]
   + simga_cut_n[56,8] + simga_cut_n[56,9] = 1
 c_y_cut_conv_combin[57]: simga_cut_n[57,0] + simga_cut_n[57,1]
   + simga_cut_n[57,2] + simga_cut_n[57,3] + simga_cut_n[57,4]
   + simga_cut_n[57,5] + simga_cut_n[57,6] + simga_cut_n[57,7]
   + simga_cut_n[57,8] + simga_cut_n[57,9] = 1
 c_y_cut_conv_combin[58]: simga_cut_n[58,0] + simga_cut_n[58,1]
   + simga_cut_n[58,2] + simga_cut_n[58,3] + simga_cut_n[58,4]
   + simga_cut_n[58,5] + simga_cut_n[58,6] + simga_cut_n[58,7]
   + simga_cut_n[58,8] + simga_cut_n[58,9] = 1
 c_y_cut_conv_combin[59]: simga_cut_n[59,0] + simga_cut_n[59,1]
   + simga_cut_n[59,2] + simga_cut_n[59,3] + simga_cut_n[59,4]
   + simga_cut_n[59,5] + simga_cut_n[59,6] + simga_cut_n[59,7]
   + simga_cut_n[59,8] + simga_cut_n[59,9] = 1
 c_y_cut_conv_combin[60]: simga_cut_n[60,0] + simga_cut_n[60,1]
   + simga_cut_n[60,2] + simga_cut_n[60,3] + simga_cut_n[60,4]
   + simga_cut_n[60,5] + simga_cut_n[60,6] + simga_cut_n[60,7]
   + simga_cut_n[60,8] + simga_cut_n[60,9] = 1
 c_y_cut_conv_combin[61]: simga_cut_n[61,0] + simga_cut_n[61,1]
   + simga_cut_n[61,2] + simga_cut_n[61,3] + simga_cut_n[61,4]
   + simga_cut_n[61,5] + simga_cut_n[61,6] + simga_cut_n[61,7]
   + simga_cut_n[61,8] + simga_cut_n[61,9] = 1
 c_y_cut_conv_combin[62]: simga_cut_n[62,0] + simga_cut_n[62,1]
   + simga_cut_n[62,2] + simga_cut_n[62,3] + simga_cut_n[62,4]
   + simga_cut_n[62,5] + simga_cut_n[62,6] + simga_cut_n[62,7]
   + simga_cut_n[62,8] + simga_cut_n[62,9] = 1
 c_y_cut_conv_combin[63]: simga_cut_n[63,0] + simga_cut_n[63,1]
   + simga_cut_n[63,2] + simga_cut_n[63,3] + simga_cut_n[63,4]
   + simga_cut_n[63,5] + simga_cut_n[63,6] + simga_cut_n[63,7]
   + simga_cut_n[63,8] + simga_cut_n[63,9] = 1
 c_y_cut_conv_combin[64]: simga_cut_n[64,0] + simga_cut_n[64,1]
   + simga_cut_n[64,2] + simga_cut_n[64,3] + simga_cut_n[64,4]
   + simga_cut_n[64,5] + simga_cut_n[64,6] + simga_cut_n[64,7]
   + simga_cut_n[64,8] + simga_cut_n[64,9] = 1
 c_y_cut_conv_combin[65]: simga_cut_n[65,0] + simga_cut_n[65,1]
   + simga_cut_n[65,2] + simga_cut_n[65,3] + simga_cut_n[65,4]
   + simga_cut_n[65,5] + simga_cut_n[65,6] + simga_cut_n[65,7]
   + simga_cut_n[65,8] + simga_cut_n[65,9] = 1
 c_y_cut_conv_combin[66]: simga_cut_n[66,0] + simga_cut_n[66,1]
   + simga_cut_n[66,2] + simga_cut_n[66,3] + simga_cut_n[66,4]
   + simga_cut_n[66,5] + simga_cut_n[66,6] + simga_cut_n[66,7]
   + simga_cut_n[66,8] + simga_cut_n[66,9] = 1
 c_y_cut_conv_combin[67]: simga_cut_n[67,0] + simga_cut_n[67,1]
   + simga_cut_n[67,2] + simga_cut_n[67,3] + simga_cut_n[67,4]
   + simga_cut_n[67,5] + simga_cut_n[67,6] + simga_cut_n[67,7]
   + simga_cut_n[67,8] + simga_cut_n[67,9] = 1
 c_y_cut_conv_combin[68]: simga_cut_n[68,0] + simga_cut_n[68,1]
   + simga_cut_n[68,2] + simga_cut_n[68,3] + simga_cut_n[68,4]
   + simga_cut_n[68,5] + simga_cut_n[68,6] + simga_cut_n[68,7]
   + simga_cut_n[68,8] + simga_cut_n[68,9] = 1
 c_y_cut_conv_combin[69]: simga_cut_n[69,0] + simga_cut_n[69,1]
   + simga_cut_n[69,2] + simga_cut_n[69,3] + simga_cut_n[69,4]
   + simga_cut_n[69,5] + simga_cut_n[69,6] + simga_cut_n[69,7]
   + simga_cut_n[69,8] + simga_cut_n[69,9] = 1
 c_y_cut_conv_combin[70]: simga_cut_n[70,0] + simga_cut_n[70,1]
   + simga_cut_n[70,2] + simga_cut_n[70,3] + simga_cut_n[70,4]
   + simga_cut_n[70,5] + simga_cut_n[70,6] + simga_cut_n[70,7]
   + simga_cut_n[70,8] + simga_cut_n[70,9] = 1
 c_y_cut_conv_combin[71]: simga_cut_n[71,0] + simga_cut_n[71,1]
   + simga_cut_n[71,2] + simga_cut_n[71,3] + simga_cut_n[71,4]
   + simga_cut_n[71,5] + simga_cut_n[71,6] + simga_cut_n[71,7]
   + simga_cut_n[71,8] + simga_cut_n[71,9] = 1
 c_y_cut_conv_combin[72]: simga_cut_n[72,0] + simga_cut_n[72,1]
   + simga_cut_n[72,2] + simga_cut_n[72,3] + simga_cut_n[72,4]
   + simga_cut_n[72,5] + simga_cut_n[72,6] + simga_cut_n[72,7]
   + simga_cut_n[72,8] + simga_cut_n[72,9] = 1
 c_y_cut_conv_combin[73]: simga_cut_n[73,0] + simga_cut_n[73,1]
   + simga_cut_n[73,2] + simga_cut_n[73,3] + simga_cut_n[73,4]
   + simga_cut_n[73,5] + simga_cut_n[73,6] + simga_cut_n[73,7]
   + simga_cut_n[73,8] + simga_cut_n[73,9] = 1
 c_y_cut_conv_combin[74]: simga_cut_n[74,0] + simga_cut_n[74,1]
   + simga_cut_n[74,2] + simga_cut_n[74,3] + simga_cut_n[74,4]
   + simga_cut_n[74,5] + simga_cut_n[74,6] + simga_cut_n[74,7]
   + simga_cut_n[74,8] + simga_cut_n[74,9] = 1
 c_y_cut_conv_combin[75]: simga_cut_n[75,0] + simga_cut_n[75,1]
   + simga_cut_n[75,2] + simga_cut_n[75,3] + simga_cut_n[75,4]
   + simga_cut_n[75,5] + simga_cut_n[75,6] + simga_cut_n[75,7]
   + simga_cut_n[75,8] + simga_cut_n[75,9] = 1
 c_y_cut_conv_combin[76]: simga_cut_n[76,0] + simga_cut_n[76,1]
   + simga_cut_n[76,2] + simga_cut_n[76,3] + simga_cut_n[76,4]
   + simga_cut_n[76,5] + simga_cut_n[76,6] + simga_cut_n[76,7]
   + simga_cut_n[76,8] + simga_cut_n[76,9] = 1
 c_y_cut_conv_combin[77]: simga_cut_n[77,0] + simga_cut_n[77,1]
   + simga_cut_n[77,2] + simga_cut_n[77,3] + simga_cut_n[77,4]
   + simga_cut_n[77,5] + simga_cut_n[77,6] + simga_cut_n[77,7]
   + simga_cut_n[77,8] + simga_cut_n[77,9] = 1
 c_y_cut_conv_combin[78]: simga_cut_n[78,0] + simga_cut_n[78,1]
   + simga_cut_n[78,2] + simga_cut_n[78,3] + simga_cut_n[78,4]
   + simga_cut_n[78,5] + simga_cut_n[78,6] + simga_cut_n[78,7]
   + simga_cut_n[78,8] + simga_cut_n[78,9] = 1
 c_y_cut_conv_combin[79]: simga_cut_n[79,0] + simga_cut_n[79,1]
   + simga_cut_n[79,2] + simga_cut_n[79,3] + simga_cut_n[79,4]
   + simga_cut_n[79,5] + simga_cut_n[79,6] + simga_cut_n[79,7]
   + simga_cut_n[79,8] + simga_cut_n[79,9] = 1
 c_y_cut_conv_combin[80]: simga_cut_n[80,0] + simga_cut_n[80,1]
   + simga_cut_n[80,2] + simga_cut_n[80,3] + simga_cut_n[80,4]
   + simga_cut_n[80,5] + simga_cut_n[80,6] + simga_cut_n[80,7]
   + simga_cut_n[80,8] + simga_cut_n[80,9] = 1
 c_y_cut_conv_combin[81]: simga_cut_n[81,0] + simga_cut_n[81,1]
   + simga_cut_n[81,2] + simga_cut_n[81,3] + simga_cut_n[81,4]
   + simga_cut_n[81,5] + simga_cut_n[81,6] + simga_cut_n[81,7]
   + simga_cut_n[81,8] + simga_cut_n[81,9] = 1
 c_y_cut_conv_combin[82]: simga_cut_n[82,0] + simga_cut_n[82,1]
   + simga_cut_n[82,2] + simga_cut_n[82,3] + simga_cut_n[82,4]
   + simga_cut_n[82,5] + simga_cut_n[82,6] + simga_cut_n[82,7]
   + simga_cut_n[82,8] + simga_cut_n[82,9] = 1
 c_y_cut_conv_combin[83]: simga_cut_n[83,0] + simga_cut_n[83,1]
   + simga_cut_n[83,2] + simga_cut_n[83,3] + simga_cut_n[83,4]
   + simga_cut_n[83,5] + simga_cut_n[83,6] + simga_cut_n[83,7]
   + simga_cut_n[83,8] + simga_cut_n[83,9] = 1
 c_y_cut_conv_combin[84]: simga_cut_n[84,0] + simga_cut_n[84,1]
   + simga_cut_n[84,2] + simga_cut_n[84,3] + simga_cut_n[84,4]
   + simga_cut_n[84,5] + simga_cut_n[84,6] + simga_cut_n[84,7]
   + simga_cut_n[84,8] + simga_cut_n[84,9] = 1
 c_y_cut_conv_combin[85]: simga_cut_n[85,0] + simga_cut_n[85,1]
   + simga_cut_n[85,2] + simga_cut_n[85,3] + simga_cut_n[85,4]
   + simga_cut_n[85,5] + simga_cut_n[85,6] + simga_cut_n[85,7]
   + simga_cut_n[85,8] + simga_cut_n[85,9] = 1
 c_y_cut_conv_combin[86]: simga_cut_n[86,0] + simga_cut_n[86,1]
   + simga_cut_n[86,2] + simga_cut_n[86,3] + simga_cut_n[86,4]
   + simga_cut_n[86,5] + simga_cut_n[86,6] + simga_cut_n[86,7]
   + simga_cut_n[86,8] + simga_cut_n[86,9] = 1
 c_y_cut_conv_combin[87]: simga_cut_n[87,0] + simga_cut_n[87,1]
   + simga_cut_n[87,2] + simga_cut_n[87,3] + simga_cut_n[87,4]
   + simga_cut_n[87,5] + simga_cut_n[87,6] + simga_cut_n[87,7]
   + simga_cut_n[87,8] + simga_cut_n[87,9] = 1
 c_y_cut_conv_combin[88]: simga_cut_n[88,0] + simga_cut_n[88,1]
   + simga_cut_n[88,2] + simga_cut_n[88,3] + simga_cut_n[88,4]
   + simga_cut_n[88,5] + simga_cut_n[88,6] + simga_cut_n[88,7]
   + simga_cut_n[88,8] + simga_cut_n[88,9] = 1
 c_y_cut_conv_combin[89]: simga_cut_n[89,0] + simga_cut_n[89,1]
   + simga_cut_n[89,2] + simga_cut_n[89,3] + simga_cut_n[89,4]
   + simga_cut_n[89,5] + simga_cut_n[89,6] + simga_cut_n[89,7]
   + simga_cut_n[89,8] + simga_cut_n[89,9] = 1
 c_y_cut_conv_combin[90]: simga_cut_n[90,0] + simga_cut_n[90,1]
   + simga_cut_n[90,2] + simga_cut_n[90,3] + simga_cut_n[90,4]
   + simga_cut_n[90,5] + simga_cut_n[90,6] + simga_cut_n[90,7]
   + simga_cut_n[90,8] + simga_cut_n[90,9] = 1
 c_y_cut_conv_combin[91]: simga_cut_n[91,0] + simga_cut_n[91,1]
   + simga_cut_n[91,2] + simga_cut_n[91,3] + simga_cut_n[91,4]
   + simga_cut_n[91,5] + simga_cut_n[91,6] + simga_cut_n[91,7]
   + simga_cut_n[91,8] + simga_cut_n[91,9] = 1
 c_y_cut_conv_combin[92]: simga_cut_n[92,0] + simga_cut_n[92,1]
   + simga_cut_n[92,2] + simga_cut_n[92,3] + simga_cut_n[92,4]
   + simga_cut_n[92,5] + simga_cut_n[92,6] + simga_cut_n[92,7]
   + simga_cut_n[92,8] + simga_cut_n[92,9] = 1
 c_y_cut_conv_combin[93]: simga_cut_n[93,0] + simga_cut_n[93,1]
   + simga_cut_n[93,2] + simga_cut_n[93,3] + simga_cut_n[93,4]
   + simga_cut_n[93,5] + simga_cut_n[93,6] + simga_cut_n[93,7]
   + simga_cut_n[93,8] + simga_cut_n[93,9] = 1
 c_y_cut_conv_combin[94]: simga_cut_n[94,0] + simga_cut_n[94,1]
   + simga_cut_n[94,2] + simga_cut_n[94,3] + simga_cut_n[94,4]
   + simga_cut_n[94,5] + simga_cut_n[94,6] + simga_cut_n[94,7]
   + simga_cut_n[94,8] + simga_cut_n[94,9] = 1
 c_y_cut_conv_combin[95]: simga_cut_n[95,0] + simga_cut_n[95,1]
   + simga_cut_n[95,2] + simga_cut_n[95,3] + simga_cut_n[95,4]
   + simga_cut_n[95,5] + simga_cut_n[95,6] + simga_cut_n[95,7]
   + simga_cut_n[95,8] + simga_cut_n[95,9] = 1
 c_y_add_conv_combin[0]: simga_add_n[0,0] + simga_add_n[0,1]
   + simga_add_n[0,2] + simga_add_n[0,3] + simga_add_n[0,4]
   + simga_add_n[0,5] + simga_add_n[0,6] + simga_add_n[0,7]
   + simga_add_n[0,8] + simga_add_n[0,9] = 1
 c_y_add_conv_combin[1]: simga_add_n[1,0] + simga_add_n[1,1]
   + simga_add_n[1,2] + simga_add_n[1,3] + simga_add_n[1,4]
   + simga_add_n[1,5] + simga_add_n[1,6] + simga_add_n[1,7]
   + simga_add_n[1,8] + simga_add_n[1,9] = 1
 c_y_add_conv_combin[2]: simga_add_n[2,0] + simga_add_n[2,1]
   + simga_add_n[2,2] + simga_add_n[2,3] + simga_add_n[2,4]
   + simga_add_n[2,5] + simga_add_n[2,6] + simga_add_n[2,7]
   + simga_add_n[2,8] + simga_add_n[2,9] = 1
 c_y_add_conv_combin[3]: simga_add_n[3,0] + simga_add_n[3,1]
   + simga_add_n[3,2] + simga_add_n[3,3] + simga_add_n[3,4]
   + simga_add_n[3,5] + simga_add_n[3,6] + simga_add_n[3,7]
   + simga_add_n[3,8] + simga_add_n[3,9] = 1
 c_y_add_conv_combin[4]: simga_add_n[4,0] + simga_add_n[4,1]
   + simga_add_n[4,2] + simga_add_n[4,3] + simga_add_n[4,4]
   + simga_add_n[4,5] + simga_add_n[4,6] + simga_add_n[4,7]
   + simga_add_n[4,8] + simga_add_n[4,9] = 1
 c_y_add_conv_combin[5]: simga_add_n[5,0] + simga_add_n[5,1]
   + simga_add_n[5,2] + simga_add_n[5,3] + simga_add_n[5,4]
   + simga_add_n[5,5] + simga_add_n[5,6] + simga_add_n[5,7]
   + simga_add_n[5,8] + simga_add_n[5,9] = 1
 c_y_add_conv_combin[6]: simga_add_n[6,0] + simga_add_n[6,1]
   + simga_add_n[6,2] + simga_add_n[6,3] + simga_add_n[6,4]
   + simga_add_n[6,5] + simga_add_n[6,6] + simga_add_n[6,7]
   + simga_add_n[6,8] + simga_add_n[6,9] = 1
 c_y_add_conv_combin[7]: simga_add_n[7,0] + simga_add_n[7,1]
   + simga_add_n[7,2] + simga_add_n[7,3] + simga_add_n[7,4]
   + simga_add_n[7,5] + simga_add_n[7,6] + simga_add_n[7,7]
   + simga_add_n[7,8] + simga_add_n[7,9] = 1
 c_y_add_conv_combin[8]: simga_add_n[8,0] + simga_add_n[8,1]
   + simga_add_n[8,2] + simga_add_n[8,3] + simga_add_n[8,4]
   + simga_add_n[8,5] + simga_add_n[8,6] + simga_add_n[8,7]
   + simga_add_n[8,8] + simga_add_n[8,9] = 1
 c_y_add_conv_combin[9]: simga_add_n[9,0] + simga_add_n[9,1]
   + simga_add_n[9,2] + simga_add_n[9,3] + simga_add_n[9,4]
   + simga_add_n[9,5] + simga_add_n[9,6] + simga_add_n[9,7]
   + simga_add_n[9,8] + simga_add_n[9,9] = 1
 c_y_add_conv_combin[10]: simga_add_n[10,0] + simga_add_n[10,1]
   + simga_add_n[10,2] + simga_add_n[10,3] + simga_add_n[10,4]
   + simga_add_n[10,5] + simga_add_n[10,6] + simga_add_n[10,7]
   + simga_add_n[10,8] + simga_add_n[10,9] = 1
 c_y_add_conv_combin[11]: simga_add_n[11,0] + simga_add_n[11,1]
   + simga_add_n[11,2] + simga_add_n[11,3] + simga_add_n[11,4]
   + simga_add_n[11,5] + simga_add_n[11,6] + simga_add_n[11,7]
   + simga_add_n[11,8] + simga_add_n[11,9] = 1
 c_y_add_conv_combin[12]: simga_add_n[12,0] + simga_add_n[12,1]
   + simga_add_n[12,2] + simga_add_n[12,3] + simga_add_n[12,4]
   + simga_add_n[12,5] + simga_add_n[12,6] + simga_add_n[12,7]
   + simga_add_n[12,8] + simga_add_n[12,9] = 1
 c_y_add_conv_combin[13]: simga_add_n[13,0] + simga_add_n[13,1]
   + simga_add_n[13,2] + simga_add_n[13,3] + simga_add_n[13,4]
   + simga_add_n[13,5] + simga_add_n[13,6] + simga_add_n[13,7]
   + simga_add_n[13,8] + simga_add_n[13,9] = 1
 c_y_add_conv_combin[14]: simga_add_n[14,0] + simga_add_n[14,1]
   + simga_add_n[14,2] + simga_add_n[14,3] + simga_add_n[14,4]
   + simga_add_n[14,5] + simga_add_n[14,6] + simga_add_n[14,7]
   + simga_add_n[14,8] + simga_add_n[14,9] = 1
 c_y_add_conv_combin[15]: simga_add_n[15,0] + simga_add_n[15,1]
   + simga_add_n[15,2] + simga_add_n[15,3] + simga_add_n[15,4]
   + simga_add_n[15,5] + simga_add_n[15,6] + simga_add_n[15,7]
   + simga_add_n[15,8] + simga_add_n[15,9] = 1
 c_y_add_conv_combin[16]: simga_add_n[16,0] + simga_add_n[16,1]
   + simga_add_n[16,2] + simga_add_n[16,3] + simga_add_n[16,4]
   + simga_add_n[16,5] + simga_add_n[16,6] + simga_add_n[16,7]
   + simga_add_n[16,8] + simga_add_n[16,9] = 1
 c_y_add_conv_combin[17]: simga_add_n[17,0] + simga_add_n[17,1]
   + simga_add_n[17,2] + simga_add_n[17,3] + simga_add_n[17,4]
   + simga_add_n[17,5] + simga_add_n[17,6] + simga_add_n[17,7]
   + simga_add_n[17,8] + simga_add_n[17,9] = 1
 c_y_add_conv_combin[18]: simga_add_n[18,0] + simga_add_n[18,1]
   + simga_add_n[18,2] + simga_add_n[18,3] + simga_add_n[18,4]
   + simga_add_n[18,5] + simga_add_n[18,6] + simga_add_n[18,7]
   + simga_add_n[18,8] + simga_add_n[18,9] = 1
 c_y_add_conv_combin[19]: simga_add_n[19,0] + simga_add_n[19,1]
   + simga_add_n[19,2] + simga_add_n[19,3] + simga_add_n[19,4]
   + simga_add_n[19,5] + simga_add_n[19,6] + simga_add_n[19,7]
   + simga_add_n[19,8] + simga_add_n[19,9] = 1
 c_y_add_conv_combin[20]: simga_add_n[20,0] + simga_add_n[20,1]
   + simga_add_n[20,2] + simga_add_n[20,3] + simga_add_n[20,4]
   + simga_add_n[20,5] + simga_add_n[20,6] + simga_add_n[20,7]
   + simga_add_n[20,8] + simga_add_n[20,9] = 1
 c_y_add_conv_combin[21]: simga_add_n[21,0] + simga_add_n[21,1]
   + simga_add_n[21,2] + simga_add_n[21,3] + simga_add_n[21,4]
   + simga_add_n[21,5] + simga_add_n[21,6] + simga_add_n[21,7]
   + simga_add_n[21,8] + simga_add_n[21,9] = 1
 c_y_add_conv_combin[22]: simga_add_n[22,0] + simga_add_n[22,1]
   + simga_add_n[22,2] + simga_add_n[22,3] + simga_add_n[22,4]
   + simga_add_n[22,5] + simga_add_n[22,6] + simga_add_n[22,7]
   + simga_add_n[22,8] + simga_add_n[22,9] = 1
 c_y_add_conv_combin[23]: simga_add_n[23,0] + simga_add_n[23,1]
   + simga_add_n[23,2] + simga_add_n[23,3] + simga_add_n[23,4]
   + simga_add_n[23,5] + simga_add_n[23,6] + simga_add_n[23,7]
   + simga_add_n[23,8] + simga_add_n[23,9] = 1
 c_y_add_conv_combin[24]: simga_add_n[24,0] + simga_add_n[24,1]
   + simga_add_n[24,2] + simga_add_n[24,3] + simga_add_n[24,4]
   + simga_add_n[24,5] + simga_add_n[24,6] + simga_add_n[24,7]
   + simga_add_n[24,8] + simga_add_n[24,9] = 1
 c_y_add_conv_combin[25]: simga_add_n[25,0] + simga_add_n[25,1]
   + simga_add_n[25,2] + simga_add_n[25,3] + simga_add_n[25,4]
   + simga_add_n[25,5] + simga_add_n[25,6] + simga_add_n[25,7]
   + simga_add_n[25,8] + simga_add_n[25,9] = 1
 c_y_add_conv_combin[26]: simga_add_n[26,0] + simga_add_n[26,1]
   + simga_add_n[26,2] + simga_add_n[26,3] + simga_add_n[26,4]
   + simga_add_n[26,5] + simga_add_n[26,6] + simga_add_n[26,7]
   + simga_add_n[26,8] + simga_add_n[26,9] = 1
 c_y_add_conv_combin[27]: simga_add_n[27,0] + simga_add_n[27,1]
   + simga_add_n[27,2] + simga_add_n[27,3] + simga_add_n[27,4]
   + simga_add_n[27,5] + simga_add_n[27,6] + simga_add_n[27,7]
   + simga_add_n[27,8] + simga_add_n[27,9] = 1
 c_y_add_conv_combin[28]: simga_add_n[28,0] + simga_add_n[28,1]
   + simga_add_n[28,2] + simga_add_n[28,3] + simga_add_n[28,4]
   + simga_add_n[28,5] + simga_add_n[28,6] + simga_add_n[28,7]
   + simga_add_n[28,8] + simga_add_n[28,9] = 1
 c_y_add_conv_combin[29]: simga_add_n[29,0] + simga_add_n[29,1]
   + simga_add_n[29,2] + simga_add_n[29,3] + simga_add_n[29,4]
   + simga_add_n[29,5] + simga_add_n[29,6] + simga_add_n[29,7]
   + simga_add_n[29,8] + simga_add_n[29,9] = 1
 c_y_add_conv_combin[30]: simga_add_n[30,0] + simga_add_n[30,1]
   + simga_add_n[30,2] + simga_add_n[30,3] + simga_add_n[30,4]
   + simga_add_n[30,5] + simga_add_n[30,6] + simga_add_n[30,7]
   + simga_add_n[30,8] + simga_add_n[30,9] = 1
 c_y_add_conv_combin[31]: simga_add_n[31,0] + simga_add_n[31,1]
   + simga_add_n[31,2] + simga_add_n[31,3] + simga_add_n[31,4]
   + simga_add_n[31,5] + simga_add_n[31,6] + simga_add_n[31,7]
   + simga_add_n[31,8] + simga_add_n[31,9] = 1
 c_y_add_conv_combin[32]: simga_add_n[32,0] + simga_add_n[32,1]
   + simga_add_n[32,2] + simga_add_n[32,3] + simga_add_n[32,4]
   + simga_add_n[32,5] + simga_add_n[32,6] + simga_add_n[32,7]
   + simga_add_n[32,8] + simga_add_n[32,9] = 1
 c_y_add_conv_combin[33]: simga_add_n[33,0] + simga_add_n[33,1]
   + simga_add_n[33,2] + simga_add_n[33,3] + simga_add_n[33,4]
   + simga_add_n[33,5] + simga_add_n[33,6] + simga_add_n[33,7]
   + simga_add_n[33,8] + simga_add_n[33,9] = 1
 c_y_add_conv_combin[34]: simga_add_n[34,0] + simga_add_n[34,1]
   + simga_add_n[34,2] + simga_add_n[34,3] + simga_add_n[34,4]
   + simga_add_n[34,5] + simga_add_n[34,6] + simga_add_n[34,7]
   + simga_add_n[34,8] + simga_add_n[34,9] = 1
 c_y_add_conv_combin[35]: simga_add_n[35,0] + simga_add_n[35,1]
   + simga_add_n[35,2] + simga_add_n[35,3] + simga_add_n[35,4]
   + simga_add_n[35,5] + simga_add_n[35,6] + simga_add_n[35,7]
   + simga_add_n[35,8] + simga_add_n[35,9] = 1
 c_y_add_conv_combin[36]: simga_add_n[36,0] + simga_add_n[36,1]
   + simga_add_n[36,2] + simga_add_n[36,3] + simga_add_n[36,4]
   + simga_add_n[36,5] + simga_add_n[36,6] + simga_add_n[36,7]
   + simga_add_n[36,8] + simga_add_n[36,9] = 1
 c_y_add_conv_combin[37]: simga_add_n[37,0] + simga_add_n[37,1]
   + simga_add_n[37,2] + simga_add_n[37,3] + simga_add_n[37,4]
   + simga_add_n[37,5] + simga_add_n[37,6] + simga_add_n[37,7]
   + simga_add_n[37,8] + simga_add_n[37,9] = 1
 c_y_add_conv_combin[38]: simga_add_n[38,0] + simga_add_n[38,1]
   + simga_add_n[38,2] + simga_add_n[38,3] + simga_add_n[38,4]
   + simga_add_n[38,5] + simga_add_n[38,6] + simga_add_n[38,7]
   + simga_add_n[38,8] + simga_add_n[38,9] = 1
 c_y_add_conv_combin[39]: simga_add_n[39,0] + simga_add_n[39,1]
   + simga_add_n[39,2] + simga_add_n[39,3] + simga_add_n[39,4]
   + simga_add_n[39,5] + simga_add_n[39,6] + simga_add_n[39,7]
   + simga_add_n[39,8] + simga_add_n[39,9] = 1
 c_y_add_conv_combin[40]: simga_add_n[40,0] + simga_add_n[40,1]
   + simga_add_n[40,2] + simga_add_n[40,3] + simga_add_n[40,4]
   + simga_add_n[40,5] + simga_add_n[40,6] + simga_add_n[40,7]
   + simga_add_n[40,8] + simga_add_n[40,9] = 1
 c_y_add_conv_combin[41]: simga_add_n[41,0] + simga_add_n[41,1]
   + simga_add_n[41,2] + simga_add_n[41,3] + simga_add_n[41,4]
   + simga_add_n[41,5] + simga_add_n[41,6] + simga_add_n[41,7]
   + simga_add_n[41,8] + simga_add_n[41,9] = 1
 c_y_add_conv_combin[42]: simga_add_n[42,0] + simga_add_n[42,1]
   + simga_add_n[42,2] + simga_add_n[42,3] + simga_add_n[42,4]
   + simga_add_n[42,5] + simga_add_n[42,6] + simga_add_n[42,7]
   + simga_add_n[42,8] + simga_add_n[42,9] = 1
 c_y_add_conv_combin[43]: simga_add_n[43,0] + simga_add_n[43,1]
   + simga_add_n[43,2] + simga_add_n[43,3] + simga_add_n[43,4]
   + simga_add_n[43,5] + simga_add_n[43,6] + simga_add_n[43,7]
   + simga_add_n[43,8] + simga_add_n[43,9] = 1
 c_y_add_conv_combin[44]: simga_add_n[44,0] + simga_add_n[44,1]
   + simga_add_n[44,2] + simga_add_n[44,3] + simga_add_n[44,4]
   + simga_add_n[44,5] + simga_add_n[44,6] + simga_add_n[44,7]
   + simga_add_n[44,8] + simga_add_n[44,9] = 1
 c_y_add_conv_combin[45]: simga_add_n[45,0] + simga_add_n[45,1]
   + simga_add_n[45,2] + simga_add_n[45,3] + simga_add_n[45,4]
   + simga_add_n[45,5] + simga_add_n[45,6] + simga_add_n[45,7]
   + simga_add_n[45,8] + simga_add_n[45,9] = 1
 c_y_add_conv_combin[46]: simga_add_n[46,0] + simga_add_n[46,1]
   + simga_add_n[46,2] + simga_add_n[46,3] + simga_add_n[46,4]
   + simga_add_n[46,5] + simga_add_n[46,6] + simga_add_n[46,7]
   + simga_add_n[46,8] + simga_add_n[46,9] = 1
 c_y_add_conv_combin[47]: simga_add_n[47,0] + simga_add_n[47,1]
   + simga_add_n[47,2] + simga_add_n[47,3] + simga_add_n[47,4]
   + simga_add_n[47,5] + simga_add_n[47,6] + simga_add_n[47,7]
   + simga_add_n[47,8] + simga_add_n[47,9] = 1
 c_y_add_conv_combin[48]: simga_add_n[48,0] + simga_add_n[48,1]
   + simga_add_n[48,2] + simga_add_n[48,3] + simga_add_n[48,4]
   + simga_add_n[48,5] + simga_add_n[48,6] + simga_add_n[48,7]
   + simga_add_n[48,8] + simga_add_n[48,9] = 1
 c_y_add_conv_combin[49]: simga_add_n[49,0] + simga_add_n[49,1]
   + simga_add_n[49,2] + simga_add_n[49,3] + simga_add_n[49,4]
   + simga_add_n[49,5] + simga_add_n[49,6] + simga_add_n[49,7]
   + simga_add_n[49,8] + simga_add_n[49,9] = 1
 c_y_add_conv_combin[50]: simga_add_n[50,0] + simga_add_n[50,1]
   + simga_add_n[50,2] + simga_add_n[50,3] + simga_add_n[50,4]
   + simga_add_n[50,5] + simga_add_n[50,6] + simga_add_n[50,7]
   + simga_add_n[50,8] + simga_add_n[50,9] = 1
 c_y_add_conv_combin[51]: simga_add_n[51,0] + simga_add_n[51,1]
   + simga_add_n[51,2] + simga_add_n[51,3] + simga_add_n[51,4]
   + simga_add_n[51,5] + simga_add_n[51,6] + simga_add_n[51,7]
   + simga_add_n[51,8] + simga_add_n[51,9] = 1
 c_y_add_conv_combin[52]: simga_add_n[52,0] + simga_add_n[52,1]
   + simga_add_n[52,2] + simga_add_n[52,3] + simga_add_n[52,4]
   + simga_add_n[52,5] + simga_add_n[52,6] + simga_add_n[52,7]
   + simga_add_n[52,8] + simga_add_n[52,9] = 1
 c_y_add_conv_combin[53]: simga_add_n[53,0] + simga_add_n[53,1]
   + simga_add_n[53,2] + simga_add_n[53,3] + simga_add_n[53,4]
   + simga_add_n[53,5] + simga_add_n[53,6] + simga_add_n[53,7]
   + simga_add_n[53,8] + simga_add_n[53,9] = 1
 c_y_add_conv_combin[54]: simga_add_n[54,0] + simga_add_n[54,1]
   + simga_add_n[54,2] + simga_add_n[54,3] + simga_add_n[54,4]
   + simga_add_n[54,5] + simga_add_n[54,6] + simga_add_n[54,7]
   + simga_add_n[54,8] + simga_add_n[54,9] = 1
 c_y_add_conv_combin[55]: simga_add_n[55,0] + simga_add_n[55,1]
   + simga_add_n[55,2] + simga_add_n[55,3] + simga_add_n[55,4]
   + simga_add_n[55,5] + simga_add_n[55,6] + simga_add_n[55,7]
   + simga_add_n[55,8] + simga_add_n[55,9] = 1
 c_y_add_conv_combin[56]: simga_add_n[56,0] + simga_add_n[56,1]
   + simga_add_n[56,2] + simga_add_n[56,3] + simga_add_n[56,4]
   + simga_add_n[56,5] + simga_add_n[56,6] + simga_add_n[56,7]
   + simga_add_n[56,8] + simga_add_n[56,9] = 1
 c_y_add_conv_combin[57]: simga_add_n[57,0] + simga_add_n[57,1]
   + simga_add_n[57,2] + simga_add_n[57,3] + simga_add_n[57,4]
   + simga_add_n[57,5] + simga_add_n[57,6] + simga_add_n[57,7]
   + simga_add_n[57,8] + simga_add_n[57,9] = 1
 c_y_add_conv_combin[58]: simga_add_n[58,0] + simga_add_n[58,1]
   + simga_add_n[58,2] + simga_add_n[58,3] + simga_add_n[58,4]
   + simga_add_n[58,5] + simga_add_n[58,6] + simga_add_n[58,7]
   + simga_add_n[58,8] + simga_add_n[58,9] = 1
 c_y_add_conv_combin[59]: simga_add_n[59,0] + simga_add_n[59,1]
   + simga_add_n[59,2] + simga_add_n[59,3] + simga_add_n[59,4]
   + simga_add_n[59,5] + simga_add_n[59,6] + simga_add_n[59,7]
   + simga_add_n[59,8] + simga_add_n[59,9] = 1
 c_y_add_conv_combin[60]: simga_add_n[60,0] + simga_add_n[60,1]
   + simga_add_n[60,2] + simga_add_n[60,3] + simga_add_n[60,4]
   + simga_add_n[60,5] + simga_add_n[60,6] + simga_add_n[60,7]
   + simga_add_n[60,8] + simga_add_n[60,9] = 1
 c_y_add_conv_combin[61]: simga_add_n[61,0] + simga_add_n[61,1]
   + simga_add_n[61,2] + simga_add_n[61,3] + simga_add_n[61,4]
   + simga_add_n[61,5] + simga_add_n[61,6] + simga_add_n[61,7]
   + simga_add_n[61,8] + simga_add_n[61,9] = 1
 c_y_add_conv_combin[62]: simga_add_n[62,0] + simga_add_n[62,1]
   + simga_add_n[62,2] + simga_add_n[62,3] + simga_add_n[62,4]
   + simga_add_n[62,5] + simga_add_n[62,6] + simga_add_n[62,7]
   + simga_add_n[62,8] + simga_add_n[62,9] = 1
 c_y_add_conv_combin[63]: simga_add_n[63,0] + simga_add_n[63,1]
   + simga_add_n[63,2] + simga_add_n[63,3] + simga_add_n[63,4]
   + simga_add_n[63,5] + simga_add_n[63,6] + simga_add_n[63,7]
   + simga_add_n[63,8] + simga_add_n[63,9] = 1
 c_y_add_conv_combin[64]: simga_add_n[64,0] + simga_add_n[64,1]
   + simga_add_n[64,2] + simga_add_n[64,3] + simga_add_n[64,4]
   + simga_add_n[64,5] + simga_add_n[64,6] + simga_add_n[64,7]
   + simga_add_n[64,8] + simga_add_n[64,9] = 1
 c_y_add_conv_combin[65]: simga_add_n[65,0] + simga_add_n[65,1]
   + simga_add_n[65,2] + simga_add_n[65,3] + simga_add_n[65,4]
   + simga_add_n[65,5] + simga_add_n[65,6] + simga_add_n[65,7]
   + simga_add_n[65,8] + simga_add_n[65,9] = 1
 c_y_add_conv_combin[66]: simga_add_n[66,0] + simga_add_n[66,1]
   + simga_add_n[66,2] + simga_add_n[66,3] + simga_add_n[66,4]
   + simga_add_n[66,5] + simga_add_n[66,6] + simga_add_n[66,7]
   + simga_add_n[66,8] + simga_add_n[66,9] = 1
 c_y_add_conv_combin[67]: simga_add_n[67,0] + simga_add_n[67,1]
   + simga_add_n[67,2] + simga_add_n[67,3] + simga_add_n[67,4]
   + simga_add_n[67,5] + simga_add_n[67,6] + simga_add_n[67,7]
   + simga_add_n[67,8] + simga_add_n[67,9] = 1
 c_y_add_conv_combin[68]: simga_add_n[68,0] + simga_add_n[68,1]
   + simga_add_n[68,2] + simga_add_n[68,3] + simga_add_n[68,4]
   + simga_add_n[68,5] + simga_add_n[68,6] + simga_add_n[68,7]
   + simga_add_n[68,8] + simga_add_n[68,9] = 1
 c_y_add_conv_combin[69]: simga_add_n[69,0] + simga_add_n[69,1]
   + simga_add_n[69,2] + simga_add_n[69,3] + simga_add_n[69,4]
   + simga_add_n[69,5] + simga_add_n[69,6] + simga_add_n[69,7]
   + simga_add_n[69,8] + simga_add_n[69,9] = 1
 c_y_add_conv_combin[70]: simga_add_n[70,0] + simga_add_n[70,1]
   + simga_add_n[70,2] + simga_add_n[70,3] + simga_add_n[70,4]
   + simga_add_n[70,5] + simga_add_n[70,6] + simga_add_n[70,7]
   + simga_add_n[70,8] + simga_add_n[70,9] = 1
 c_y_add_conv_combin[71]: simga_add_n[71,0] + simga_add_n[71,1]
   + simga_add_n[71,2] + simga_add_n[71,3] + simga_add_n[71,4]
   + simga_add_n[71,5] + simga_add_n[71,6] + simga_add_n[71,7]
   + simga_add_n[71,8] + simga_add_n[71,9] = 1
 c_y_add_conv_combin[72]: simga_add_n[72,0] + simga_add_n[72,1]
   + simga_add_n[72,2] + simga_add_n[72,3] + simga_add_n[72,4]
   + simga_add_n[72,5] + simga_add_n[72,6] + simga_add_n[72,7]
   + simga_add_n[72,8] + simga_add_n[72,9] = 1
 c_y_add_conv_combin[73]: simga_add_n[73,0] + simga_add_n[73,1]
   + simga_add_n[73,2] + simga_add_n[73,3] + simga_add_n[73,4]
   + simga_add_n[73,5] + simga_add_n[73,6] + simga_add_n[73,7]
   + simga_add_n[73,8] + simga_add_n[73,9] = 1
 c_y_add_conv_combin[74]: simga_add_n[74,0] + simga_add_n[74,1]
   + simga_add_n[74,2] + simga_add_n[74,3] + simga_add_n[74,4]
   + simga_add_n[74,5] + simga_add_n[74,6] + simga_add_n[74,7]
   + simga_add_n[74,8] + simga_add_n[74,9] = 1
 c_y_add_conv_combin[75]: simga_add_n[75,0] + simga_add_n[75,1]
   + simga_add_n[75,2] + simga_add_n[75,3] + simga_add_n[75,4]
   + simga_add_n[75,5] + simga_add_n[75,6] + simga_add_n[75,7]
   + simga_add_n[75,8] + simga_add_n[75,9] = 1
 c_y_add_conv_combin[76]: simga_add_n[76,0] + simga_add_n[76,1]
   + simga_add_n[76,2] + simga_add_n[76,3] + simga_add_n[76,4]
   + simga_add_n[76,5] + simga_add_n[76,6] + simga_add_n[76,7]
   + simga_add_n[76,8] + simga_add_n[76,9] = 1
 c_y_add_conv_combin[77]: simga_add_n[77,0] + simga_add_n[77,1]
   + simga_add_n[77,2] + simga_add_n[77,3] + simga_add_n[77,4]
   + simga_add_n[77,5] + simga_add_n[77,6] + simga_add_n[77,7]
   + simga_add_n[77,8] + simga_add_n[77,9] = 1
 c_y_add_conv_combin[78]: simga_add_n[78,0] + simga_add_n[78,1]
   + simga_add_n[78,2] + simga_add_n[78,3] + simga_add_n[78,4]
   + simga_add_n[78,5] + simga_add_n[78,6] + simga_add_n[78,7]
   + simga_add_n[78,8] + simga_add_n[78,9] = 1
 c_y_add_conv_combin[79]: simga_add_n[79,0] + simga_add_n[79,1]
   + simga_add_n[79,2] + simga_add_n[79,3] + simga_add_n[79,4]
   + simga_add_n[79,5] + simga_add_n[79,6] + simga_add_n[79,7]
   + simga_add_n[79,8] + simga_add_n[79,9] = 1
 c_y_add_conv_combin[80]: simga_add_n[80,0] + simga_add_n[80,1]
   + simga_add_n[80,2] + simga_add_n[80,3] + simga_add_n[80,4]
   + simga_add_n[80,5] + simga_add_n[80,6] + simga_add_n[80,7]
   + simga_add_n[80,8] + simga_add_n[80,9] = 1
 c_y_add_conv_combin[81]: simga_add_n[81,0] + simga_add_n[81,1]
   + simga_add_n[81,2] + simga_add_n[81,3] + simga_add_n[81,4]
   + simga_add_n[81,5] + simga_add_n[81,6] + simga_add_n[81,7]
   + simga_add_n[81,8] + simga_add_n[81,9] = 1
 c_y_add_conv_combin[82]: simga_add_n[82,0] + simga_add_n[82,1]
   + simga_add_n[82,2] + simga_add_n[82,3] + simga_add_n[82,4]
   + simga_add_n[82,5] + simga_add_n[82,6] + simga_add_n[82,7]
   + simga_add_n[82,8] + simga_add_n[82,9] = 1
 c_y_add_conv_combin[83]: simga_add_n[83,0] + simga_add_n[83,1]
   + simga_add_n[83,2] + simga_add_n[83,3] + simga_add_n[83,4]
   + simga_add_n[83,5] + simga_add_n[83,6] + simga_add_n[83,7]
   + simga_add_n[83,8] + simga_add_n[83,9] = 1
 c_y_add_conv_combin[84]: simga_add_n[84,0] + simga_add_n[84,1]
   + simga_add_n[84,2] + simga_add_n[84,3] + simga_add_n[84,4]
   + simga_add_n[84,5] + simga_add_n[84,6] + simga_add_n[84,7]
   + simga_add_n[84,8] + simga_add_n[84,9] = 1
 c_y_add_conv_combin[85]: simga_add_n[85,0] + simga_add_n[85,1]
   + simga_add_n[85,2] + simga_add_n[85,3] + simga_add_n[85,4]
   + simga_add_n[85,5] + simga_add_n[85,6] + simga_add_n[85,7]
   + simga_add_n[85,8] + simga_add_n[85,9] = 1
 c_y_add_conv_combin[86]: simga_add_n[86,0] + simga_add_n[86,1]
   + simga_add_n[86,2] + simga_add_n[86,3] + simga_add_n[86,4]
   + simga_add_n[86,5] + simga_add_n[86,6] + simga_add_n[86,7]
   + simga_add_n[86,8] + simga_add_n[86,9] = 1
 c_y_add_conv_combin[87]: simga_add_n[87,0] + simga_add_n[87,1]
   + simga_add_n[87,2] + simga_add_n[87,3] + simga_add_n[87,4]
   + simga_add_n[87,5] + simga_add_n[87,6] + simga_add_n[87,7]
   + simga_add_n[87,8] + simga_add_n[87,9] = 1
 c_y_add_conv_combin[88]: simga_add_n[88,0] + simga_add_n[88,1]
   + simga_add_n[88,2] + simga_add_n[88,3] + simga_add_n[88,4]
   + simga_add_n[88,5] + simga_add_n[88,6] + simga_add_n[88,7]
   + simga_add_n[88,8] + simga_add_n[88,9] = 1
 c_y_add_conv_combin[89]: simga_add_n[89,0] + simga_add_n[89,1]
   + simga_add_n[89,2] + simga_add_n[89,3] + simga_add_n[89,4]
   + simga_add_n[89,5] + simga_add_n[89,6] + simga_add_n[89,7]
   + simga_add_n[89,8] + simga_add_n[89,9] = 1
 c_y_add_conv_combin[90]: simga_add_n[90,0] + simga_add_n[90,1]
   + simga_add_n[90,2] + simga_add_n[90,3] + simga_add_n[90,4]
   + simga_add_n[90,5] + simga_add_n[90,6] + simga_add_n[90,7]
   + simga_add_n[90,8] + simga_add_n[90,9] = 1
 c_y_add_conv_combin[91]: simga_add_n[91,0] + simga_add_n[91,1]
   + simga_add_n[91,2] + simga_add_n[91,3] + simga_add_n[91,4]
   + simga_add_n[91,5] + simga_add_n[91,6] + simga_add_n[91,7]
   + simga_add_n[91,8] + simga_add_n[91,9] = 1
 c_y_add_conv_combin[92]: simga_add_n[92,0] + simga_add_n[92,1]
   + simga_add_n[92,2] + simga_add_n[92,3] + simga_add_n[92,4]
   + simga_add_n[92,5] + simga_add_n[92,6] + simga_add_n[92,7]
   + simga_add_n[92,8] + simga_add_n[92,9] = 1
 c_y_add_conv_combin[93]: simga_add_n[93,0] + simga_add_n[93,1]
   + simga_add_n[93,2] + simga_add_n[93,3] + simga_add_n[93,4]
   + simga_add_n[93,5] + simga_add_n[93,6] + simga_add_n[93,7]
   + simga_add_n[93,8] + simga_add_n[93,9] = 1
 c_y_add_conv_combin[94]: simga_add_n[94,0] + simga_add_n[94,1]
   + simga_add_n[94,2] + simga_add_n[94,3] + simga_add_n[94,4]
   + simga_add_n[94,5] + simga_add_n[94,6] + simga_add_n[94,7]
   + simga_add_n[94,8] + simga_add_n[94,9] = 1
 c_y_add_conv_combin[95]: simga_add_n[95,0] + simga_add_n[95,1]
   + simga_add_n[95,2] + simga_add_n[95,3] + simga_add_n[95,4]
   + simga_add_n[95,5] + simga_add_n[95,6] + simga_add_n[95,7]
   + simga_add_n[95,8] + simga_add_n[95,9] = 1
Bounds
End
