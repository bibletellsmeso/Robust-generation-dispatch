\ Model planner_MILP_gurobi
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  x_cost_fuel[0] + x_cost_fuel[1] + x_cost_fuel[2] + x_cost_fuel[3]
   + x_cost_fuel[4] + x_cost_fuel[5] + x_cost_fuel[6] + x_cost_fuel[7]
   + x_cost_fuel[8] + x_cost_fuel[9] + x_cost_fuel[10] + x_cost_fuel[11]
   + x_cost_fuel[12] + x_cost_fuel[13] + x_cost_fuel[14] + x_cost_fuel[15]
   + x_cost_fuel[16] + x_cost_fuel[17] + x_cost_fuel[18] + x_cost_fuel[19]
   + x_cost_fuel[20] + x_cost_fuel[21] + x_cost_fuel[22] + x_cost_fuel[23]
   + x_cost_fuel[24] + x_cost_fuel[25] + x_cost_fuel[26] + x_cost_fuel[27]
   + x_cost_fuel[28] + x_cost_fuel[29] + x_cost_fuel[30] + x_cost_fuel[31]
   + x_cost_fuel[32] + x_cost_fuel[33] + x_cost_fuel[34] + x_cost_fuel[35]
   + x_cost_fuel[36] + x_cost_fuel[37] + x_cost_fuel[38] + x_cost_fuel[39]
   + x_cost_fuel[40] + x_cost_fuel[41] + x_cost_fuel[42] + x_cost_fuel[43]
   + x_cost_fuel[44] + x_cost_fuel[45] + x_cost_fuel[46] + x_cost_fuel[47]
   + x_cost_fuel[48] + x_cost_fuel[49] + x_cost_fuel[50] + x_cost_fuel[51]
   + x_cost_fuel[52] + x_cost_fuel[53] + x_cost_fuel[54] + x_cost_fuel[55]
   + x_cost_fuel[56] + x_cost_fuel[57] + x_cost_fuel[58] + x_cost_fuel[59]
   + x_cost_fuel[60] + x_cost_fuel[61] + x_cost_fuel[62] + x_cost_fuel[63]
   + x_cost_fuel[64] + x_cost_fuel[65] + x_cost_fuel[66] + x_cost_fuel[67]
   + x_cost_fuel[68] + x_cost_fuel[69] + x_cost_fuel[70] + x_cost_fuel[71]
   + x_cost_fuel[72] + x_cost_fuel[73] + x_cost_fuel[74] + x_cost_fuel[75]
   + x_cost_fuel[76] + x_cost_fuel[77] + x_cost_fuel[78] + x_cost_fuel[79]
   + x_cost_fuel[80] + x_cost_fuel[81] + x_cost_fuel[82] + x_cost_fuel[83]
   + x_cost_fuel[84] + x_cost_fuel[85] + x_cost_fuel[86] + x_cost_fuel[87]
   + x_cost_fuel[88] + x_cost_fuel[89] + x_cost_fuel[90] + x_cost_fuel[91]
   + x_cost_fuel[92] + x_cost_fuel[93] + x_cost_fuel[94] + x_cost_fuel[95]
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
   + x_cost_ESS[0] + x_cost_ESS[1] + x_cost_ESS[2] + x_cost_ESS[3]
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
   + x_cost_cut[0] + x_cost_cut[1] + x_cost_cut[2] + x_cost_cut[3]
   + x_cost_cut[4] + x_cost_cut[5] + x_cost_cut[6] + x_cost_cut[7]
   + x_cost_cut[8] + x_cost_cut[9] + x_cost_cut[10] + x_cost_cut[11]
   + x_cost_cut[12] + x_cost_cut[13] + x_cost_cut[14] + x_cost_cut[15]
   + x_cost_cut[16] + x_cost_cut[17] + x_cost_cut[18] + x_cost_cut[19]
   + x_cost_cut[20] + x_cost_cut[21] + x_cost_cut[22] + x_cost_cut[23]
   + x_cost_cut[24] + x_cost_cut[25] + x_cost_cut[26] + x_cost_cut[27]
   + x_cost_cut[28] + x_cost_cut[29] + x_cost_cut[30] + x_cost_cut[31]
   + x_cost_cut[32] + x_cost_cut[33] + x_cost_cut[34] + x_cost_cut[35]
   + x_cost_cut[36] + x_cost_cut[37] + x_cost_cut[38] + x_cost_cut[39]
   + x_cost_cut[40] + x_cost_cut[41] + x_cost_cut[42] + x_cost_cut[43]
   + x_cost_cut[44] + x_cost_cut[45] + x_cost_cut[46] + x_cost_cut[47]
   + x_cost_cut[48] + x_cost_cut[49] + x_cost_cut[50] + x_cost_cut[51]
   + x_cost_cut[52] + x_cost_cut[53] + x_cost_cut[54] + x_cost_cut[55]
   + x_cost_cut[56] + x_cost_cut[57] + x_cost_cut[58] + x_cost_cut[59]
   + x_cost_cut[60] + x_cost_cut[61] + x_cost_cut[62] + x_cost_cut[63]
   + x_cost_cut[64] + x_cost_cut[65] + x_cost_cut[66] + x_cost_cut[67]
   + x_cost_cut[68] + x_cost_cut[69] + x_cost_cut[70] + x_cost_cut[71]
   + x_cost_cut[72] + x_cost_cut[73] + x_cost_cut[74] + x_cost_cut[75]
   + x_cost_cut[76] + x_cost_cut[77] + x_cost_cut[78] + x_cost_cut[79]
   + x_cost_cut[80] + x_cost_cut[81] + x_cost_cut[82] + x_cost_cut[83]
   + x_cost_cut[84] + x_cost_cut[85] + x_cost_cut[86] + x_cost_cut[87]
   + x_cost_cut[88] + x_cost_cut[89] + x_cost_cut[90] + x_cost_cut[91]
   + x_cost_cut[92] + x_cost_cut[93] + x_cost_cut[94] + x_cost_cut[95]
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
   + y_cost_cut[0] + y_cost_cut[1] + y_cost_cut[2] + y_cost_cut[3]
   + y_cost_cut[4] + y_cost_cut[5] + y_cost_cut[6] + y_cost_cut[7]
   + y_cost_cut[8] + y_cost_cut[9] + y_cost_cut[10] + y_cost_cut[11]
   + y_cost_cut[12] + y_cost_cut[13] + y_cost_cut[14] + y_cost_cut[15]
   + y_cost_cut[16] + y_cost_cut[17] + y_cost_cut[18] + y_cost_cut[19]
   + y_cost_cut[20] + y_cost_cut[21] + y_cost_cut[22] + y_cost_cut[23]
   + y_cost_cut[24] + y_cost_cut[25] + y_cost_cut[26] + y_cost_cut[27]
   + y_cost_cut[28] + y_cost_cut[29] + y_cost_cut[30] + y_cost_cut[31]
   + y_cost_cut[32] + y_cost_cut[33] + y_cost_cut[34] + y_cost_cut[35]
   + y_cost_cut[36] + y_cost_cut[37] + y_cost_cut[38] + y_cost_cut[39]
   + y_cost_cut[40] + y_cost_cut[41] + y_cost_cut[42] + y_cost_cut[43]
   + y_cost_cut[44] + y_cost_cut[45] + y_cost_cut[46] + y_cost_cut[47]
   + y_cost_cut[48] + y_cost_cut[49] + y_cost_cut[50] + y_cost_cut[51]
   + y_cost_cut[52] + y_cost_cut[53] + y_cost_cut[54] + y_cost_cut[55]
   + y_cost_cut[56] + y_cost_cut[57] + y_cost_cut[58] + y_cost_cut[59]
   + y_cost_cut[60] + y_cost_cut[61] + y_cost_cut[62] + y_cost_cut[63]
   + y_cost_cut[64] + y_cost_cut[65] + y_cost_cut[66] + y_cost_cut[67]
   + y_cost_cut[68] + y_cost_cut[69] + y_cost_cut[70] + y_cost_cut[71]
   + y_cost_cut[72] + y_cost_cut[73] + y_cost_cut[74] + y_cost_cut[75]
   + y_cost_cut[76] + y_cost_cut[77] + y_cost_cut[78] + y_cost_cut[79]
   + y_cost_cut[80] + y_cost_cut[81] + y_cost_cut[82] + y_cost_cut[83]
   + y_cost_cut[84] + y_cost_cut[85] + y_cost_cut[86] + y_cost_cut[87]
   + y_cost_cut[88] + y_cost_cut[89] + y_cost_cut[90] + y_cost_cut[91]
   + y_cost_cut[92] + y_cost_cut[93] + y_cost_cut[94] + y_cost_cut[95]
   + y_cost_add[0] + y_cost_add[1] + y_cost_add[2] + y_cost_add[3]
   + y_cost_add[4] + y_cost_add[5] + y_cost_add[6] + y_cost_add[7]
   + y_cost_add[8] + y_cost_add[9] + y_cost_add[10] + y_cost_add[11]
   + y_cost_add[12] + y_cost_add[13] + y_cost_add[14] + y_cost_add[15]
   + y_cost_add[16] + y_cost_add[17] + y_cost_add[18] + y_cost_add[19]
   + y_cost_add[20] + y_cost_add[21] + y_cost_add[22] + y_cost_add[23]
   + y_cost_add[24] + y_cost_add[25] + y_cost_add[26] + y_cost_add[27]
   + y_cost_add[28] + y_cost_add[29] + y_cost_add[30] + y_cost_add[31]
   + y_cost_add[32] + y_cost_add[33] + y_cost_add[34] + y_cost_add[35]
   + y_cost_add[36] + y_cost_add[37] + y_cost_add[38] + y_cost_add[39]
   + y_cost_add[40] + y_cost_add[41] + y_cost_add[42] + y_cost_add[43]
   + y_cost_add[44] + y_cost_add[45] + y_cost_add[46] + y_cost_add[47]
   + y_cost_add[48] + y_cost_add[49] + y_cost_add[50] + y_cost_add[51]
   + y_cost_add[52] + y_cost_add[53] + y_cost_add[54] + y_cost_add[55]
   + y_cost_add[56] + y_cost_add[57] + y_cost_add[58] + y_cost_add[59]
   + y_cost_add[60] + y_cost_add[61] + y_cost_add[62] + y_cost_add[63]
   + y_cost_add[64] + y_cost_add[65] + y_cost_add[66] + y_cost_add[67]
   + y_cost_add[68] + y_cost_add[69] + y_cost_add[70] + y_cost_add[71]
   + y_cost_add[72] + y_cost_add[73] + y_cost_add[74] + y_cost_add[75]
   + y_cost_add[76] + y_cost_add[77] + y_cost_add[78] + y_cost_add[79]
   + y_cost_add[80] + y_cost_add[81] + y_cost_add[82] + y_cost_add[83]
   + y_cost_add[84] + y_cost_add[85] + y_cost_add[86] + y_cost_add[87]
   + y_cost_add[88] + y_cost_add[89] + y_cost_add[90] + y_cost_add[91]
   + y_cost_add[92] + y_cost_add[93] + y_cost_add[94] + y_cost_add[95]
Subject To
 c_x_min[0]: - x[0] <= -150
 c_x_min[1]: - x[1] <= -150
 c_x_min[2]: - x[2] <= -150
 c_x_min[3]: - x[3] <= -150
 c_x_min[4]: - x[4] <= -150
 c_x_min[5]: - x[5] <= -150
 c_x_min[6]: - x[6] <= -150
 c_x_min[7]: - x[7] <= -150
 c_x_min[8]: - x[8] <= -150
 c_x_min[9]: - x[9] <= -150
 c_x_min[10]: - x[10] <= -150
 c_x_min[11]: - x[11] <= -150
 c_x_min[12]: - x[12] <= -150
 c_x_min[13]: - x[13] <= -150
 c_x_min[14]: - x[14] <= -150
 c_x_min[15]: - x[15] <= -150
 c_x_min[16]: - x[16] <= -150
 c_x_min[17]: - x[17] <= -150
 c_x_min[18]: - x[18] <= -150
 c_x_min[19]: - x[19] <= -150
 c_x_min[20]: - x[20] <= -150
 c_x_min[21]: - x[21] <= -150
 c_x_min[22]: - x[22] <= -150
 c_x_min[23]: - x[23] <= -150
 c_x_min[24]: - x[24] <= -150
 c_x_min[25]: - x[25] <= -150
 c_x_min[26]: - x[26] <= -150
 c_x_min[27]: - x[27] <= -150
 c_x_min[28]: - x[28] <= -150
 c_x_min[29]: - x[29] <= -150
 c_x_min[30]: - x[30] <= -150
 c_x_min[31]: - x[31] <= -150
 c_x_min[32]: - x[32] <= -150
 c_x_min[33]: - x[33] <= -150
 c_x_min[34]: - x[34] <= -150
 c_x_min[35]: - x[35] <= -150
 c_x_min[36]: - x[36] <= -150
 c_x_min[37]: - x[37] <= -150
 c_x_min[38]: - x[38] <= -150
 c_x_min[39]: - x[39] <= -150
 c_x_min[40]: - x[40] <= -150
 c_x_min[41]: - x[41] <= -150
 c_x_min[42]: - x[42] <= -150
 c_x_min[43]: - x[43] <= -150
 c_x_min[44]: - x[44] <= -150
 c_x_min[45]: - x[45] <= -150
 c_x_min[46]: - x[46] <= -150
 c_x_min[47]: - x[47] <= -150
 c_x_min[48]: - x[48] <= -150
 c_x_min[49]: - x[49] <= -150
 c_x_min[50]: - x[50] <= -150
 c_x_min[51]: - x[51] <= -150
 c_x_min[52]: - x[52] <= -150
 c_x_min[53]: - x[53] <= -150
 c_x_min[54]: - x[54] <= -150
 c_x_min[55]: - x[55] <= -150
 c_x_min[56]: - x[56] <= -150
 c_x_min[57]: - x[57] <= -150
 c_x_min[58]: - x[58] <= -150
 c_x_min[59]: - x[59] <= -150
 c_x_min[60]: - x[60] <= -150
 c_x_min[61]: - x[61] <= -150
 c_x_min[62]: - x[62] <= -150
 c_x_min[63]: - x[63] <= -150
 c_x_min[64]: - x[64] <= -150
 c_x_min[65]: - x[65] <= -150
 c_x_min[66]: - x[66] <= -150
 c_x_min[67]: - x[67] <= -150
 c_x_min[68]: - x[68] <= -150
 c_x_min[69]: - x[69] <= -150
 c_x_min[70]: - x[70] <= -150
 c_x_min[71]: - x[71] <= -150
 c_x_min[72]: - x[72] <= -150
 c_x_min[73]: - x[73] <= -150
 c_x_min[74]: - x[74] <= -150
 c_x_min[75]: - x[75] <= -150
 c_x_min[76]: - x[76] <= -150
 c_x_min[77]: - x[77] <= -150
 c_x_min[78]: - x[78] <= -150
 c_x_min[79]: - x[79] <= -150
 c_x_min[80]: - x[80] <= -150
 c_x_min[81]: - x[81] <= -150
 c_x_min[82]: - x[82] <= -150
 c_x_min[83]: - x[83] <= -150
 c_x_min[84]: - x[84] <= -150
 c_x_min[85]: - x[85] <= -150
 c_x_min[86]: - x[86] <= -150
 c_x_min[87]: - x[87] <= -150
 c_x_min[88]: - x[88] <= -150
 c_x_min[89]: - x[89] <= -150
 c_x_min[90]: - x[90] <= -150
 c_x_min[91]: - x[91] <= -150
 c_x_min[92]: - x[92] <= -150
 c_x_min[93]: - x[93] <= -150
 c_x_min[94]: - x[94] <= -150
 c_x_min[95]: - x[95] <= -150
 c_x_max[0]: x[0] <= 750
 c_x_max[1]: x[1] <= 750
 c_x_max[2]: x[2] <= 750
 c_x_max[3]: x[3] <= 750
 c_x_max[4]: x[4] <= 750
 c_x_max[5]: x[5] <= 750
 c_x_max[6]: x[6] <= 750
 c_x_max[7]: x[7] <= 750
 c_x_max[8]: x[8] <= 750
 c_x_max[9]: x[9] <= 750
 c_x_max[10]: x[10] <= 750
 c_x_max[11]: x[11] <= 750
 c_x_max[12]: x[12] <= 750
 c_x_max[13]: x[13] <= 750
 c_x_max[14]: x[14] <= 750
 c_x_max[15]: x[15] <= 750
 c_x_max[16]: x[16] <= 750
 c_x_max[17]: x[17] <= 750
 c_x_max[18]: x[18] <= 750
 c_x_max[19]: x[19] <= 750
 c_x_max[20]: x[20] <= 750
 c_x_max[21]: x[21] <= 750
 c_x_max[22]: x[22] <= 750
 c_x_max[23]: x[23] <= 750
 c_x_max[24]: x[24] <= 750
 c_x_max[25]: x[25] <= 750
 c_x_max[26]: x[26] <= 750
 c_x_max[27]: x[27] <= 750
 c_x_max[28]: x[28] <= 750
 c_x_max[29]: x[29] <= 750
 c_x_max[30]: x[30] <= 750
 c_x_max[31]: x[31] <= 750
 c_x_max[32]: x[32] <= 750
 c_x_max[33]: x[33] <= 750
 c_x_max[34]: x[34] <= 750
 c_x_max[35]: x[35] <= 750
 c_x_max[36]: x[36] <= 750
 c_x_max[37]: x[37] <= 750
 c_x_max[38]: x[38] <= 750
 c_x_max[39]: x[39] <= 750
 c_x_max[40]: x[40] <= 750
 c_x_max[41]: x[41] <= 750
 c_x_max[42]: x[42] <= 750
 c_x_max[43]: x[43] <= 750
 c_x_max[44]: x[44] <= 750
 c_x_max[45]: x[45] <= 750
 c_x_max[46]: x[46] <= 750
 c_x_max[47]: x[47] <= 750
 c_x_max[48]: x[48] <= 750
 c_x_max[49]: x[49] <= 750
 c_x_max[50]: x[50] <= 750
 c_x_max[51]: x[51] <= 750
 c_x_max[52]: x[52] <= 750
 c_x_max[53]: x[53] <= 750
 c_x_max[54]: x[54] <= 750
 c_x_max[55]: x[55] <= 750
 c_x_max[56]: x[56] <= 750
 c_x_max[57]: x[57] <= 750
 c_x_max[58]: x[58] <= 750
 c_x_max[59]: x[59] <= 750
 c_x_max[60]: x[60] <= 750
 c_x_max[61]: x[61] <= 750
 c_x_max[62]: x[62] <= 750
 c_x_max[63]: x[63] <= 750
 c_x_max[64]: x[64] <= 750
 c_x_max[65]: x[65] <= 750
 c_x_max[66]: x[66] <= 750
 c_x_max[67]: x[67] <= 750
 c_x_max[68]: x[68] <= 750
 c_x_max[69]: x[69] <= 750
 c_x_max[70]: x[70] <= 750
 c_x_max[71]: x[71] <= 750
 c_x_max[72]: x[72] <= 750
 c_x_max[73]: x[73] <= 750
 c_x_max[74]: x[74] <= 750
 c_x_max[75]: x[75] <= 750
 c_x_max[76]: x[76] <= 750
 c_x_max[77]: x[77] <= 750
 c_x_max[78]: x[78] <= 750
 c_x_max[79]: x[79] <= 750
 c_x_max[80]: x[80] <= 750
 c_x_max[81]: x[81] <= 750
 c_x_max[82]: x[82] <= 750
 c_x_max[83]: x[83] <= 750
 c_x_max[84]: x[84] <= 750
 c_x_max[85]: x[85] <= 750
 c_x_max[86]: x[86] <= 750
 c_x_max[87]: x[87] <= 750
 c_x_max[88]: x[88] <= 750
 c_x_max[89]: x[89] <= 750
 c_x_max[90]: x[90] <= 750
 c_x_max[91]: x[91] <= 750
 c_x_max[92]: x[92] <= 750
 c_x_max[93]: x[93] <= 750
 c_x_max[94]: x[94] <= 750
 c_x_max[95]: x[95] <= 750
 c_DG_RAMP1[1]: - x[0] + x[1] + x_pos[1] + x_neg[0] <= 100
 c_DG_RAMP1[2]: - x[1] + x[2] + x_pos[2] + x_neg[1] <= 100
 c_DG_RAMP1[3]: - x[2] + x[3] + x_pos[3] + x_neg[2] <= 100
 c_DG_RAMP1[4]: - x[3] + x[4] + x_pos[4] + x_neg[3] <= 100
 c_DG_RAMP1[5]: - x[4] + x[5] + x_pos[5] + x_neg[4] <= 100
 c_DG_RAMP1[6]: - x[5] + x[6] + x_pos[6] + x_neg[5] <= 100
 c_DG_RAMP1[7]: - x[6] + x[7] + x_pos[7] + x_neg[6] <= 100
 c_DG_RAMP1[8]: - x[7] + x[8] + x_pos[8] + x_neg[7] <= 100
 c_DG_RAMP1[9]: - x[8] + x[9] + x_pos[9] + x_neg[8] <= 100
 c_DG_RAMP1[10]: - x[9] + x[10] + x_pos[10] + x_neg[9] <= 100
 c_DG_RAMP1[11]: - x[10] + x[11] + x_pos[11] + x_neg[10] <= 100
 c_DG_RAMP1[12]: - x[11] + x[12] + x_pos[12] + x_neg[11] <= 100
 c_DG_RAMP1[13]: - x[12] + x[13] + x_pos[13] + x_neg[12] <= 100
 c_DG_RAMP1[14]: - x[13] + x[14] + x_pos[14] + x_neg[13] <= 100
 c_DG_RAMP1[15]: - x[14] + x[15] + x_pos[15] + x_neg[14] <= 100
 c_DG_RAMP1[16]: - x[15] + x[16] + x_pos[16] + x_neg[15] <= 100
 c_DG_RAMP1[17]: - x[16] + x[17] + x_pos[17] + x_neg[16] <= 100
 c_DG_RAMP1[18]: - x[17] + x[18] + x_pos[18] + x_neg[17] <= 100
 c_DG_RAMP1[19]: - x[18] + x[19] + x_pos[19] + x_neg[18] <= 100
 c_DG_RAMP1[20]: - x[19] + x[20] + x_pos[20] + x_neg[19] <= 100
 c_DG_RAMP1[21]: - x[20] + x[21] + x_pos[21] + x_neg[20] <= 100
 c_DG_RAMP1[22]: - x[21] + x[22] + x_pos[22] + x_neg[21] <= 100
 c_DG_RAMP1[23]: - x[22] + x[23] + x_pos[23] + x_neg[22] <= 100
 c_DG_RAMP1[24]: - x[23] + x[24] + x_pos[24] + x_neg[23] <= 100
 c_DG_RAMP1[25]: - x[24] + x[25] + x_pos[25] + x_neg[24] <= 100
 c_DG_RAMP1[26]: - x[25] + x[26] + x_pos[26] + x_neg[25] <= 100
 c_DG_RAMP1[27]: - x[26] + x[27] + x_pos[27] + x_neg[26] <= 100
 c_DG_RAMP1[28]: - x[27] + x[28] + x_pos[28] + x_neg[27] <= 100
 c_DG_RAMP1[29]: - x[28] + x[29] + x_pos[29] + x_neg[28] <= 100
 c_DG_RAMP1[30]: - x[29] + x[30] + x_pos[30] + x_neg[29] <= 100
 c_DG_RAMP1[31]: - x[30] + x[31] + x_pos[31] + x_neg[30] <= 100
 c_DG_RAMP1[32]: - x[31] + x[32] + x_pos[32] + x_neg[31] <= 100
 c_DG_RAMP1[33]: - x[32] + x[33] + x_pos[33] + x_neg[32] <= 100
 c_DG_RAMP1[34]: - x[33] + x[34] + x_pos[34] + x_neg[33] <= 100
 c_DG_RAMP1[35]: - x[34] + x[35] + x_pos[35] + x_neg[34] <= 100
 c_DG_RAMP1[36]: - x[35] + x[36] + x_pos[36] + x_neg[35] <= 100
 c_DG_RAMP1[37]: - x[36] + x[37] + x_pos[37] + x_neg[36] <= 100
 c_DG_RAMP1[38]: - x[37] + x[38] + x_pos[38] + x_neg[37] <= 100
 c_DG_RAMP1[39]: - x[38] + x[39] + x_pos[39] + x_neg[38] <= 100
 c_DG_RAMP1[40]: - x[39] + x[40] + x_pos[40] + x_neg[39] <= 100
 c_DG_RAMP1[41]: - x[40] + x[41] + x_pos[41] + x_neg[40] <= 100
 c_DG_RAMP1[42]: - x[41] + x[42] + x_pos[42] + x_neg[41] <= 100
 c_DG_RAMP1[43]: - x[42] + x[43] + x_pos[43] + x_neg[42] <= 100
 c_DG_RAMP1[44]: - x[43] + x[44] + x_pos[44] + x_neg[43] <= 100
 c_DG_RAMP1[45]: - x[44] + x[45] + x_pos[45] + x_neg[44] <= 100
 c_DG_RAMP1[46]: - x[45] + x[46] + x_pos[46] + x_neg[45] <= 100
 c_DG_RAMP1[47]: - x[46] + x[47] + x_pos[47] + x_neg[46] <= 100
 c_DG_RAMP1[48]: - x[47] + x[48] + x_pos[48] + x_neg[47] <= 100
 c_DG_RAMP1[49]: - x[48] + x[49] + x_pos[49] + x_neg[48] <= 100
 c_DG_RAMP1[50]: - x[49] + x[50] + x_pos[50] + x_neg[49] <= 100
 c_DG_RAMP1[51]: - x[50] + x[51] + x_pos[51] + x_neg[50] <= 100
 c_DG_RAMP1[52]: - x[51] + x[52] + x_pos[52] + x_neg[51] <= 100
 c_DG_RAMP1[53]: - x[52] + x[53] + x_pos[53] + x_neg[52] <= 100
 c_DG_RAMP1[54]: - x[53] + x[54] + x_pos[54] + x_neg[53] <= 100
 c_DG_RAMP1[55]: - x[54] + x[55] + x_pos[55] + x_neg[54] <= 100
 c_DG_RAMP1[56]: - x[55] + x[56] + x_pos[56] + x_neg[55] <= 100
 c_DG_RAMP1[57]: - x[56] + x[57] + x_pos[57] + x_neg[56] <= 100
 c_DG_RAMP1[58]: - x[57] + x[58] + x_pos[58] + x_neg[57] <= 100
 c_DG_RAMP1[59]: - x[58] + x[59] + x_pos[59] + x_neg[58] <= 100
 c_DG_RAMP1[60]: - x[59] + x[60] + x_pos[60] + x_neg[59] <= 100
 c_DG_RAMP1[61]: - x[60] + x[61] + x_pos[61] + x_neg[60] <= 100
 c_DG_RAMP1[62]: - x[61] + x[62] + x_pos[62] + x_neg[61] <= 100
 c_DG_RAMP1[63]: - x[62] + x[63] + x_pos[63] + x_neg[62] <= 100
 c_DG_RAMP1[64]: - x[63] + x[64] + x_pos[64] + x_neg[63] <= 100
 c_DG_RAMP1[65]: - x[64] + x[65] + x_pos[65] + x_neg[64] <= 100
 c_DG_RAMP1[66]: - x[65] + x[66] + x_pos[66] + x_neg[65] <= 100
 c_DG_RAMP1[67]: - x[66] + x[67] + x_pos[67] + x_neg[66] <= 100
 c_DG_RAMP1[68]: - x[67] + x[68] + x_pos[68] + x_neg[67] <= 100
 c_DG_RAMP1[69]: - x[68] + x[69] + x_pos[69] + x_neg[68] <= 100
 c_DG_RAMP1[70]: - x[69] + x[70] + x_pos[70] + x_neg[69] <= 100
 c_DG_RAMP1[71]: - x[70] + x[71] + x_pos[71] + x_neg[70] <= 100
 c_DG_RAMP1[72]: - x[71] + x[72] + x_pos[72] + x_neg[71] <= 100
 c_DG_RAMP1[73]: - x[72] + x[73] + x_pos[73] + x_neg[72] <= 100
 c_DG_RAMP1[74]: - x[73] + x[74] + x_pos[74] + x_neg[73] <= 100
 c_DG_RAMP1[75]: - x[74] + x[75] + x_pos[75] + x_neg[74] <= 100
 c_DG_RAMP1[76]: - x[75] + x[76] + x_pos[76] + x_neg[75] <= 100
 c_DG_RAMP1[77]: - x[76] + x[77] + x_pos[77] + x_neg[76] <= 100
 c_DG_RAMP1[78]: - x[77] + x[78] + x_pos[78] + x_neg[77] <= 100
 c_DG_RAMP1[79]: - x[78] + x[79] + x_pos[79] + x_neg[78] <= 100
 c_DG_RAMP1[80]: - x[79] + x[80] + x_pos[80] + x_neg[79] <= 100
 c_DG_RAMP1[81]: - x[80] + x[81] + x_pos[81] + x_neg[80] <= 100
 c_DG_RAMP1[82]: - x[81] + x[82] + x_pos[82] + x_neg[81] <= 100
 c_DG_RAMP1[83]: - x[82] + x[83] + x_pos[83] + x_neg[82] <= 100
 c_DG_RAMP1[84]: - x[83] + x[84] + x_pos[84] + x_neg[83] <= 100
 c_DG_RAMP1[85]: - x[84] + x[85] + x_pos[85] + x_neg[84] <= 100
 c_DG_RAMP1[86]: - x[85] + x[86] + x_pos[86] + x_neg[85] <= 100
 c_DG_RAMP1[87]: - x[86] + x[87] + x_pos[87] + x_neg[86] <= 100
 c_DG_RAMP1[88]: - x[87] + x[88] + x_pos[88] + x_neg[87] <= 100
 c_DG_RAMP1[89]: - x[88] + x[89] + x_pos[89] + x_neg[88] <= 100
 c_DG_RAMP1[90]: - x[89] + x[90] + x_pos[90] + x_neg[89] <= 100
 c_DG_RAMP1[91]: - x[90] + x[91] + x_pos[91] + x_neg[90] <= 100
 c_DG_RAMP1[92]: - x[91] + x[92] + x_pos[92] + x_neg[91] <= 100
 c_DG_RAMP1[93]: - x[92] + x[93] + x_pos[93] + x_neg[92] <= 100
 c_DG_RAMP1[94]: - x[93] + x[94] + x_pos[94] + x_neg[93] <= 100
 c_DG_RAMP1[95]: - x[94] + x[95] + x_pos[95] + x_neg[94] <= 100
 c_DG_RAMP4[1]: x[0] - x[1] + x_pos[0] + x_neg[1] <= 100
 c_DG_RAMP4[2]: x[1] - x[2] + x_pos[1] + x_neg[2] <= 100
 c_DG_RAMP4[3]: x[2] - x[3] + x_pos[2] + x_neg[3] <= 100
 c_DG_RAMP4[4]: x[3] - x[4] + x_pos[3] + x_neg[4] <= 100
 c_DG_RAMP4[5]: x[4] - x[5] + x_pos[4] + x_neg[5] <= 100
 c_DG_RAMP4[6]: x[5] - x[6] + x_pos[5] + x_neg[6] <= 100
 c_DG_RAMP4[7]: x[6] - x[7] + x_pos[6] + x_neg[7] <= 100
 c_DG_RAMP4[8]: x[7] - x[8] + x_pos[7] + x_neg[8] <= 100
 c_DG_RAMP4[9]: x[8] - x[9] + x_pos[8] + x_neg[9] <= 100
 c_DG_RAMP4[10]: x[9] - x[10] + x_pos[9] + x_neg[10] <= 100
 c_DG_RAMP4[11]: x[10] - x[11] + x_pos[10] + x_neg[11] <= 100
 c_DG_RAMP4[12]: x[11] - x[12] + x_pos[11] + x_neg[12] <= 100
 c_DG_RAMP4[13]: x[12] - x[13] + x_pos[12] + x_neg[13] <= 100
 c_DG_RAMP4[14]: x[13] - x[14] + x_pos[13] + x_neg[14] <= 100
 c_DG_RAMP4[15]: x[14] - x[15] + x_pos[14] + x_neg[15] <= 100
 c_DG_RAMP4[16]: x[15] - x[16] + x_pos[15] + x_neg[16] <= 100
 c_DG_RAMP4[17]: x[16] - x[17] + x_pos[16] + x_neg[17] <= 100
 c_DG_RAMP4[18]: x[17] - x[18] + x_pos[17] + x_neg[18] <= 100
 c_DG_RAMP4[19]: x[18] - x[19] + x_pos[18] + x_neg[19] <= 100
 c_DG_RAMP4[20]: x[19] - x[20] + x_pos[19] + x_neg[20] <= 100
 c_DG_RAMP4[21]: x[20] - x[21] + x_pos[20] + x_neg[21] <= 100
 c_DG_RAMP4[22]: x[21] - x[22] + x_pos[21] + x_neg[22] <= 100
 c_DG_RAMP4[23]: x[22] - x[23] + x_pos[22] + x_neg[23] <= 100
 c_DG_RAMP4[24]: x[23] - x[24] + x_pos[23] + x_neg[24] <= 100
 c_DG_RAMP4[25]: x[24] - x[25] + x_pos[24] + x_neg[25] <= 100
 c_DG_RAMP4[26]: x[25] - x[26] + x_pos[25] + x_neg[26] <= 100
 c_DG_RAMP4[27]: x[26] - x[27] + x_pos[26] + x_neg[27] <= 100
 c_DG_RAMP4[28]: x[27] - x[28] + x_pos[27] + x_neg[28] <= 100
 c_DG_RAMP4[29]: x[28] - x[29] + x_pos[28] + x_neg[29] <= 100
 c_DG_RAMP4[30]: x[29] - x[30] + x_pos[29] + x_neg[30] <= 100
 c_DG_RAMP4[31]: x[30] - x[31] + x_pos[30] + x_neg[31] <= 100
 c_DG_RAMP4[32]: x[31] - x[32] + x_pos[31] + x_neg[32] <= 100
 c_DG_RAMP4[33]: x[32] - x[33] + x_pos[32] + x_neg[33] <= 100
 c_DG_RAMP4[34]: x[33] - x[34] + x_pos[33] + x_neg[34] <= 100
 c_DG_RAMP4[35]: x[34] - x[35] + x_pos[34] + x_neg[35] <= 100
 c_DG_RAMP4[36]: x[35] - x[36] + x_pos[35] + x_neg[36] <= 100
 c_DG_RAMP4[37]: x[36] - x[37] + x_pos[36] + x_neg[37] <= 100
 c_DG_RAMP4[38]: x[37] - x[38] + x_pos[37] + x_neg[38] <= 100
 c_DG_RAMP4[39]: x[38] - x[39] + x_pos[38] + x_neg[39] <= 100
 c_DG_RAMP4[40]: x[39] - x[40] + x_pos[39] + x_neg[40] <= 100
 c_DG_RAMP4[41]: x[40] - x[41] + x_pos[40] + x_neg[41] <= 100
 c_DG_RAMP4[42]: x[41] - x[42] + x_pos[41] + x_neg[42] <= 100
 c_DG_RAMP4[43]: x[42] - x[43] + x_pos[42] + x_neg[43] <= 100
 c_DG_RAMP4[44]: x[43] - x[44] + x_pos[43] + x_neg[44] <= 100
 c_DG_RAMP4[45]: x[44] - x[45] + x_pos[44] + x_neg[45] <= 100
 c_DG_RAMP4[46]: x[45] - x[46] + x_pos[45] + x_neg[46] <= 100
 c_DG_RAMP4[47]: x[46] - x[47] + x_pos[46] + x_neg[47] <= 100
 c_DG_RAMP4[48]: x[47] - x[48] + x_pos[47] + x_neg[48] <= 100
 c_DG_RAMP4[49]: x[48] - x[49] + x_pos[48] + x_neg[49] <= 100
 c_DG_RAMP4[50]: x[49] - x[50] + x_pos[49] + x_neg[50] <= 100
 c_DG_RAMP4[51]: x[50] - x[51] + x_pos[50] + x_neg[51] <= 100
 c_DG_RAMP4[52]: x[51] - x[52] + x_pos[51] + x_neg[52] <= 100
 c_DG_RAMP4[53]: x[52] - x[53] + x_pos[52] + x_neg[53] <= 100
 c_DG_RAMP4[54]: x[53] - x[54] + x_pos[53] + x_neg[54] <= 100
 c_DG_RAMP4[55]: x[54] - x[55] + x_pos[54] + x_neg[55] <= 100
 c_DG_RAMP4[56]: x[55] - x[56] + x_pos[55] + x_neg[56] <= 100
 c_DG_RAMP4[57]: x[56] - x[57] + x_pos[56] + x_neg[57] <= 100
 c_DG_RAMP4[58]: x[57] - x[58] + x_pos[57] + x_neg[58] <= 100
 c_DG_RAMP4[59]: x[58] - x[59] + x_pos[58] + x_neg[59] <= 100
 c_DG_RAMP4[60]: x[59] - x[60] + x_pos[59] + x_neg[60] <= 100
 c_DG_RAMP4[61]: x[60] - x[61] + x_pos[60] + x_neg[61] <= 100
 c_DG_RAMP4[62]: x[61] - x[62] + x_pos[61] + x_neg[62] <= 100
 c_DG_RAMP4[63]: x[62] - x[63] + x_pos[62] + x_neg[63] <= 100
 c_DG_RAMP4[64]: x[63] - x[64] + x_pos[63] + x_neg[64] <= 100
 c_DG_RAMP4[65]: x[64] - x[65] + x_pos[64] + x_neg[65] <= 100
 c_DG_RAMP4[66]: x[65] - x[66] + x_pos[65] + x_neg[66] <= 100
 c_DG_RAMP4[67]: x[66] - x[67] + x_pos[66] + x_neg[67] <= 100
 c_DG_RAMP4[68]: x[67] - x[68] + x_pos[67] + x_neg[68] <= 100
 c_DG_RAMP4[69]: x[68] - x[69] + x_pos[68] + x_neg[69] <= 100
 c_DG_RAMP4[70]: x[69] - x[70] + x_pos[69] + x_neg[70] <= 100
 c_DG_RAMP4[71]: x[70] - x[71] + x_pos[70] + x_neg[71] <= 100
 c_DG_RAMP4[72]: x[71] - x[72] + x_pos[71] + x_neg[72] <= 100
 c_DG_RAMP4[73]: x[72] - x[73] + x_pos[72] + x_neg[73] <= 100
 c_DG_RAMP4[74]: x[73] - x[74] + x_pos[73] + x_neg[74] <= 100
 c_DG_RAMP4[75]: x[74] - x[75] + x_pos[74] + x_neg[75] <= 100
 c_DG_RAMP4[76]: x[75] - x[76] + x_pos[75] + x_neg[76] <= 100
 c_DG_RAMP4[77]: x[76] - x[77] + x_pos[76] + x_neg[77] <= 100
 c_DG_RAMP4[78]: x[77] - x[78] + x_pos[77] + x_neg[78] <= 100
 c_DG_RAMP4[79]: x[78] - x[79] + x_pos[78] + x_neg[79] <= 100
 c_DG_RAMP4[80]: x[79] - x[80] + x_pos[79] + x_neg[80] <= 100
 c_DG_RAMP4[81]: x[80] - x[81] + x_pos[80] + x_neg[81] <= 100
 c_DG_RAMP4[82]: x[81] - x[82] + x_pos[81] + x_neg[82] <= 100
 c_DG_RAMP4[83]: x[82] - x[83] + x_pos[82] + x_neg[83] <= 100
 c_DG_RAMP4[84]: x[83] - x[84] + x_pos[83] + x_neg[84] <= 100
 c_DG_RAMP4[85]: x[84] - x[85] + x_pos[84] + x_neg[85] <= 100
 c_DG_RAMP4[86]: x[85] - x[86] + x_pos[85] + x_neg[86] <= 100
 c_DG_RAMP4[87]: x[86] - x[87] + x_pos[86] + x_neg[87] <= 100
 c_DG_RAMP4[88]: x[87] - x[88] + x_pos[87] + x_neg[88] <= 100
 c_DG_RAMP4[89]: x[88] - x[89] + x_pos[88] + x_neg[89] <= 100
 c_DG_RAMP4[90]: x[89] - x[90] + x_pos[89] + x_neg[90] <= 100
 c_DG_RAMP4[91]: x[90] - x[91] + x_pos[90] + x_neg[91] <= 100
 c_DG_RAMP4[92]: x[91] - x[92] + x_pos[91] + x_neg[92] <= 100
 c_DG_RAMP4[93]: x[92] - x[93] + x_pos[92] + x_neg[93] <= 100
 c_DG_RAMP4[94]: x[93] - x[94] + x_pos[93] + x_neg[94] <= 100
 c_DG_RAMP4[95]: x[94] - x[95] + x_pos[94] + x_neg[95] <= 100
 c_xr_neg_min[0]: - x[0] + x_neg[0] <= -150
 c_xr_neg_min[1]: - x[1] + x_neg[1] <= -150
 c_xr_neg_min[2]: - x[2] + x_neg[2] <= -150
 c_xr_neg_min[3]: - x[3] + x_neg[3] <= -150
 c_xr_neg_min[4]: - x[4] + x_neg[4] <= -150
 c_xr_neg_min[5]: - x[5] + x_neg[5] <= -150
 c_xr_neg_min[6]: - x[6] + x_neg[6] <= -150
 c_xr_neg_min[7]: - x[7] + x_neg[7] <= -150
 c_xr_neg_min[8]: - x[8] + x_neg[8] <= -150
 c_xr_neg_min[9]: - x[9] + x_neg[9] <= -150
 c_xr_neg_min[10]: - x[10] + x_neg[10] <= -150
 c_xr_neg_min[11]: - x[11] + x_neg[11] <= -150
 c_xr_neg_min[12]: - x[12] + x_neg[12] <= -150
 c_xr_neg_min[13]: - x[13] + x_neg[13] <= -150
 c_xr_neg_min[14]: - x[14] + x_neg[14] <= -150
 c_xr_neg_min[15]: - x[15] + x_neg[15] <= -150
 c_xr_neg_min[16]: - x[16] + x_neg[16] <= -150
 c_xr_neg_min[17]: - x[17] + x_neg[17] <= -150
 c_xr_neg_min[18]: - x[18] + x_neg[18] <= -150
 c_xr_neg_min[19]: - x[19] + x_neg[19] <= -150
 c_xr_neg_min[20]: - x[20] + x_neg[20] <= -150
 c_xr_neg_min[21]: - x[21] + x_neg[21] <= -150
 c_xr_neg_min[22]: - x[22] + x_neg[22] <= -150
 c_xr_neg_min[23]: - x[23] + x_neg[23] <= -150
 c_xr_neg_min[24]: - x[24] + x_neg[24] <= -150
 c_xr_neg_min[25]: - x[25] + x_neg[25] <= -150
 c_xr_neg_min[26]: - x[26] + x_neg[26] <= -150
 c_xr_neg_min[27]: - x[27] + x_neg[27] <= -150
 c_xr_neg_min[28]: - x[28] + x_neg[28] <= -150
 c_xr_neg_min[29]: - x[29] + x_neg[29] <= -150
 c_xr_neg_min[30]: - x[30] + x_neg[30] <= -150
 c_xr_neg_min[31]: - x[31] + x_neg[31] <= -150
 c_xr_neg_min[32]: - x[32] + x_neg[32] <= -150
 c_xr_neg_min[33]: - x[33] + x_neg[33] <= -150
 c_xr_neg_min[34]: - x[34] + x_neg[34] <= -150
 c_xr_neg_min[35]: - x[35] + x_neg[35] <= -150
 c_xr_neg_min[36]: - x[36] + x_neg[36] <= -150
 c_xr_neg_min[37]: - x[37] + x_neg[37] <= -150
 c_xr_neg_min[38]: - x[38] + x_neg[38] <= -150
 c_xr_neg_min[39]: - x[39] + x_neg[39] <= -150
 c_xr_neg_min[40]: - x[40] + x_neg[40] <= -150
 c_xr_neg_min[41]: - x[41] + x_neg[41] <= -150
 c_xr_neg_min[42]: - x[42] + x_neg[42] <= -150
 c_xr_neg_min[43]: - x[43] + x_neg[43] <= -150
 c_xr_neg_min[44]: - x[44] + x_neg[44] <= -150
 c_xr_neg_min[45]: - x[45] + x_neg[45] <= -150
 c_xr_neg_min[46]: - x[46] + x_neg[46] <= -150
 c_xr_neg_min[47]: - x[47] + x_neg[47] <= -150
 c_xr_neg_min[48]: - x[48] + x_neg[48] <= -150
 c_xr_neg_min[49]: - x[49] + x_neg[49] <= -150
 c_xr_neg_min[50]: - x[50] + x_neg[50] <= -150
 c_xr_neg_min[51]: - x[51] + x_neg[51] <= -150
 c_xr_neg_min[52]: - x[52] + x_neg[52] <= -150
 c_xr_neg_min[53]: - x[53] + x_neg[53] <= -150
 c_xr_neg_min[54]: - x[54] + x_neg[54] <= -150
 c_xr_neg_min[55]: - x[55] + x_neg[55] <= -150
 c_xr_neg_min[56]: - x[56] + x_neg[56] <= -150
 c_xr_neg_min[57]: - x[57] + x_neg[57] <= -150
 c_xr_neg_min[58]: - x[58] + x_neg[58] <= -150
 c_xr_neg_min[59]: - x[59] + x_neg[59] <= -150
 c_xr_neg_min[60]: - x[60] + x_neg[60] <= -150
 c_xr_neg_min[61]: - x[61] + x_neg[61] <= -150
 c_xr_neg_min[62]: - x[62] + x_neg[62] <= -150
 c_xr_neg_min[63]: - x[63] + x_neg[63] <= -150
 c_xr_neg_min[64]: - x[64] + x_neg[64] <= -150
 c_xr_neg_min[65]: - x[65] + x_neg[65] <= -150
 c_xr_neg_min[66]: - x[66] + x_neg[66] <= -150
 c_xr_neg_min[67]: - x[67] + x_neg[67] <= -150
 c_xr_neg_min[68]: - x[68] + x_neg[68] <= -150
 c_xr_neg_min[69]: - x[69] + x_neg[69] <= -150
 c_xr_neg_min[70]: - x[70] + x_neg[70] <= -150
 c_xr_neg_min[71]: - x[71] + x_neg[71] <= -150
 c_xr_neg_min[72]: - x[72] + x_neg[72] <= -150
 c_xr_neg_min[73]: - x[73] + x_neg[73] <= -150
 c_xr_neg_min[74]: - x[74] + x_neg[74] <= -150
 c_xr_neg_min[75]: - x[75] + x_neg[75] <= -150
 c_xr_neg_min[76]: - x[76] + x_neg[76] <= -150
 c_xr_neg_min[77]: - x[77] + x_neg[77] <= -150
 c_xr_neg_min[78]: - x[78] + x_neg[78] <= -150
 c_xr_neg_min[79]: - x[79] + x_neg[79] <= -150
 c_xr_neg_min[80]: - x[80] + x_neg[80] <= -150
 c_xr_neg_min[81]: - x[81] + x_neg[81] <= -150
 c_xr_neg_min[82]: - x[82] + x_neg[82] <= -150
 c_xr_neg_min[83]: - x[83] + x_neg[83] <= -150
 c_xr_neg_min[84]: - x[84] + x_neg[84] <= -150
 c_xr_neg_min[85]: - x[85] + x_neg[85] <= -150
 c_xr_neg_min[86]: - x[86] + x_neg[86] <= -150
 c_xr_neg_min[87]: - x[87] + x_neg[87] <= -150
 c_xr_neg_min[88]: - x[88] + x_neg[88] <= -150
 c_xr_neg_min[89]: - x[89] + x_neg[89] <= -150
 c_xr_neg_min[90]: - x[90] + x_neg[90] <= -150
 c_xr_neg_min[91]: - x[91] + x_neg[91] <= -150
 c_xr_neg_min[92]: - x[92] + x_neg[92] <= -150
 c_xr_neg_min[93]: - x[93] + x_neg[93] <= -150
 c_xr_neg_min[94]: - x[94] + x_neg[94] <= -150
 c_xr_neg_min[95]: - x[95] + x_neg[95] <= -150
 c_xr_neg_max[0]: x[0] - x_neg[0] <= 750
 c_xr_neg_max[1]: x[1] - x_neg[1] <= 750
 c_xr_neg_max[2]: x[2] - x_neg[2] <= 750
 c_xr_neg_max[3]: x[3] - x_neg[3] <= 750
 c_xr_neg_max[4]: x[4] - x_neg[4] <= 750
 c_xr_neg_max[5]: x[5] - x_neg[5] <= 750
 c_xr_neg_max[6]: x[6] - x_neg[6] <= 750
 c_xr_neg_max[7]: x[7] - x_neg[7] <= 750
 c_xr_neg_max[8]: x[8] - x_neg[8] <= 750
 c_xr_neg_max[9]: x[9] - x_neg[9] <= 750
 c_xr_neg_max[10]: x[10] - x_neg[10] <= 750
 c_xr_neg_max[11]: x[11] - x_neg[11] <= 750
 c_xr_neg_max[12]: x[12] - x_neg[12] <= 750
 c_xr_neg_max[13]: x[13] - x_neg[13] <= 750
 c_xr_neg_max[14]: x[14] - x_neg[14] <= 750
 c_xr_neg_max[15]: x[15] - x_neg[15] <= 750
 c_xr_neg_max[16]: x[16] - x_neg[16] <= 750
 c_xr_neg_max[17]: x[17] - x_neg[17] <= 750
 c_xr_neg_max[18]: x[18] - x_neg[18] <= 750
 c_xr_neg_max[19]: x[19] - x_neg[19] <= 750
 c_xr_neg_max[20]: x[20] - x_neg[20] <= 750
 c_xr_neg_max[21]: x[21] - x_neg[21] <= 750
 c_xr_neg_max[22]: x[22] - x_neg[22] <= 750
 c_xr_neg_max[23]: x[23] - x_neg[23] <= 750
 c_xr_neg_max[24]: x[24] - x_neg[24] <= 750
 c_xr_neg_max[25]: x[25] - x_neg[25] <= 750
 c_xr_neg_max[26]: x[26] - x_neg[26] <= 750
 c_xr_neg_max[27]: x[27] - x_neg[27] <= 750
 c_xr_neg_max[28]: x[28] - x_neg[28] <= 750
 c_xr_neg_max[29]: x[29] - x_neg[29] <= 750
 c_xr_neg_max[30]: x[30] - x_neg[30] <= 750
 c_xr_neg_max[31]: x[31] - x_neg[31] <= 750
 c_xr_neg_max[32]: x[32] - x_neg[32] <= 750
 c_xr_neg_max[33]: x[33] - x_neg[33] <= 750
 c_xr_neg_max[34]: x[34] - x_neg[34] <= 750
 c_xr_neg_max[35]: x[35] - x_neg[35] <= 750
 c_xr_neg_max[36]: x[36] - x_neg[36] <= 750
 c_xr_neg_max[37]: x[37] - x_neg[37] <= 750
 c_xr_neg_max[38]: x[38] - x_neg[38] <= 750
 c_xr_neg_max[39]: x[39] - x_neg[39] <= 750
 c_xr_neg_max[40]: x[40] - x_neg[40] <= 750
 c_xr_neg_max[41]: x[41] - x_neg[41] <= 750
 c_xr_neg_max[42]: x[42] - x_neg[42] <= 750
 c_xr_neg_max[43]: x[43] - x_neg[43] <= 750
 c_xr_neg_max[44]: x[44] - x_neg[44] <= 750
 c_xr_neg_max[45]: x[45] - x_neg[45] <= 750
 c_xr_neg_max[46]: x[46] - x_neg[46] <= 750
 c_xr_neg_max[47]: x[47] - x_neg[47] <= 750
 c_xr_neg_max[48]: x[48] - x_neg[48] <= 750
 c_xr_neg_max[49]: x[49] - x_neg[49] <= 750
 c_xr_neg_max[50]: x[50] - x_neg[50] <= 750
 c_xr_neg_max[51]: x[51] - x_neg[51] <= 750
 c_xr_neg_max[52]: x[52] - x_neg[52] <= 750
 c_xr_neg_max[53]: x[53] - x_neg[53] <= 750
 c_xr_neg_max[54]: x[54] - x_neg[54] <= 750
 c_xr_neg_max[55]: x[55] - x_neg[55] <= 750
 c_xr_neg_max[56]: x[56] - x_neg[56] <= 750
 c_xr_neg_max[57]: x[57] - x_neg[57] <= 750
 c_xr_neg_max[58]: x[58] - x_neg[58] <= 750
 c_xr_neg_max[59]: x[59] - x_neg[59] <= 750
 c_xr_neg_max[60]: x[60] - x_neg[60] <= 750
 c_xr_neg_max[61]: x[61] - x_neg[61] <= 750
 c_xr_neg_max[62]: x[62] - x_neg[62] <= 750
 c_xr_neg_max[63]: x[63] - x_neg[63] <= 750
 c_xr_neg_max[64]: x[64] - x_neg[64] <= 750
 c_xr_neg_max[65]: x[65] - x_neg[65] <= 750
 c_xr_neg_max[66]: x[66] - x_neg[66] <= 750
 c_xr_neg_max[67]: x[67] - x_neg[67] <= 750
 c_xr_neg_max[68]: x[68] - x_neg[68] <= 750
 c_xr_neg_max[69]: x[69] - x_neg[69] <= 750
 c_xr_neg_max[70]: x[70] - x_neg[70] <= 750
 c_xr_neg_max[71]: x[71] - x_neg[71] <= 750
 c_xr_neg_max[72]: x[72] - x_neg[72] <= 750
 c_xr_neg_max[73]: x[73] - x_neg[73] <= 750
 c_xr_neg_max[74]: x[74] - x_neg[74] <= 750
 c_xr_neg_max[75]: x[75] - x_neg[75] <= 750
 c_xr_neg_max[76]: x[76] - x_neg[76] <= 750
 c_xr_neg_max[77]: x[77] - x_neg[77] <= 750
 c_xr_neg_max[78]: x[78] - x_neg[78] <= 750
 c_xr_neg_max[79]: x[79] - x_neg[79] <= 750
 c_xr_neg_max[80]: x[80] - x_neg[80] <= 750
 c_xr_neg_max[81]: x[81] - x_neg[81] <= 750
 c_xr_neg_max[82]: x[82] - x_neg[82] <= 750
 c_xr_neg_max[83]: x[83] - x_neg[83] <= 750
 c_xr_neg_max[84]: x[84] - x_neg[84] <= 750
 c_xr_neg_max[85]: x[85] - x_neg[85] <= 750
 c_xr_neg_max[86]: x[86] - x_neg[86] <= 750
 c_xr_neg_max[87]: x[87] - x_neg[87] <= 750
 c_xr_neg_max[88]: x[88] - x_neg[88] <= 750
 c_xr_neg_max[89]: x[89] - x_neg[89] <= 750
 c_xr_neg_max[90]: x[90] - x_neg[90] <= 750
 c_xr_neg_max[91]: x[91] - x_neg[91] <= 750
 c_xr_neg_max[92]: x[92] - x_neg[92] <= 750
 c_xr_neg_max[93]: x[93] - x_neg[93] <= 750
 c_xr_neg_max[94]: x[94] - x_neg[94] <= 750
 c_xr_neg_max[95]: x[95] - x_neg[95] <= 750
 c_xr_pos_min[0]: - x[0] - x_pos[0] <= -150
 c_xr_pos_min[1]: - x[1] - x_pos[1] <= -150
 c_xr_pos_min[2]: - x[2] - x_pos[2] <= -150
 c_xr_pos_min[3]: - x[3] - x_pos[3] <= -150
 c_xr_pos_min[4]: - x[4] - x_pos[4] <= -150
 c_xr_pos_min[5]: - x[5] - x_pos[5] <= -150
 c_xr_pos_min[6]: - x[6] - x_pos[6] <= -150
 c_xr_pos_min[7]: - x[7] - x_pos[7] <= -150
 c_xr_pos_min[8]: - x[8] - x_pos[8] <= -150
 c_xr_pos_min[9]: - x[9] - x_pos[9] <= -150
 c_xr_pos_min[10]: - x[10] - x_pos[10] <= -150
 c_xr_pos_min[11]: - x[11] - x_pos[11] <= -150
 c_xr_pos_min[12]: - x[12] - x_pos[12] <= -150
 c_xr_pos_min[13]: - x[13] - x_pos[13] <= -150
 c_xr_pos_min[14]: - x[14] - x_pos[14] <= -150
 c_xr_pos_min[15]: - x[15] - x_pos[15] <= -150
 c_xr_pos_min[16]: - x[16] - x_pos[16] <= -150
 c_xr_pos_min[17]: - x[17] - x_pos[17] <= -150
 c_xr_pos_min[18]: - x[18] - x_pos[18] <= -150
 c_xr_pos_min[19]: - x[19] - x_pos[19] <= -150
 c_xr_pos_min[20]: - x[20] - x_pos[20] <= -150
 c_xr_pos_min[21]: - x[21] - x_pos[21] <= -150
 c_xr_pos_min[22]: - x[22] - x_pos[22] <= -150
 c_xr_pos_min[23]: - x[23] - x_pos[23] <= -150
 c_xr_pos_min[24]: - x[24] - x_pos[24] <= -150
 c_xr_pos_min[25]: - x[25] - x_pos[25] <= -150
 c_xr_pos_min[26]: - x[26] - x_pos[26] <= -150
 c_xr_pos_min[27]: - x[27] - x_pos[27] <= -150
 c_xr_pos_min[28]: - x[28] - x_pos[28] <= -150
 c_xr_pos_min[29]: - x[29] - x_pos[29] <= -150
 c_xr_pos_min[30]: - x[30] - x_pos[30] <= -150
 c_xr_pos_min[31]: - x[31] - x_pos[31] <= -150
 c_xr_pos_min[32]: - x[32] - x_pos[32] <= -150
 c_xr_pos_min[33]: - x[33] - x_pos[33] <= -150
 c_xr_pos_min[34]: - x[34] - x_pos[34] <= -150
 c_xr_pos_min[35]: - x[35] - x_pos[35] <= -150
 c_xr_pos_min[36]: - x[36] - x_pos[36] <= -150
 c_xr_pos_min[37]: - x[37] - x_pos[37] <= -150
 c_xr_pos_min[38]: - x[38] - x_pos[38] <= -150
 c_xr_pos_min[39]: - x[39] - x_pos[39] <= -150
 c_xr_pos_min[40]: - x[40] - x_pos[40] <= -150
 c_xr_pos_min[41]: - x[41] - x_pos[41] <= -150
 c_xr_pos_min[42]: - x[42] - x_pos[42] <= -150
 c_xr_pos_min[43]: - x[43] - x_pos[43] <= -150
 c_xr_pos_min[44]: - x[44] - x_pos[44] <= -150
 c_xr_pos_min[45]: - x[45] - x_pos[45] <= -150
 c_xr_pos_min[46]: - x[46] - x_pos[46] <= -150
 c_xr_pos_min[47]: - x[47] - x_pos[47] <= -150
 c_xr_pos_min[48]: - x[48] - x_pos[48] <= -150
 c_xr_pos_min[49]: - x[49] - x_pos[49] <= -150
 c_xr_pos_min[50]: - x[50] - x_pos[50] <= -150
 c_xr_pos_min[51]: - x[51] - x_pos[51] <= -150
 c_xr_pos_min[52]: - x[52] - x_pos[52] <= -150
 c_xr_pos_min[53]: - x[53] - x_pos[53] <= -150
 c_xr_pos_min[54]: - x[54] - x_pos[54] <= -150
 c_xr_pos_min[55]: - x[55] - x_pos[55] <= -150
 c_xr_pos_min[56]: - x[56] - x_pos[56] <= -150
 c_xr_pos_min[57]: - x[57] - x_pos[57] <= -150
 c_xr_pos_min[58]: - x[58] - x_pos[58] <= -150
 c_xr_pos_min[59]: - x[59] - x_pos[59] <= -150
 c_xr_pos_min[60]: - x[60] - x_pos[60] <= -150
 c_xr_pos_min[61]: - x[61] - x_pos[61] <= -150
 c_xr_pos_min[62]: - x[62] - x_pos[62] <= -150
 c_xr_pos_min[63]: - x[63] - x_pos[63] <= -150
 c_xr_pos_min[64]: - x[64] - x_pos[64] <= -150
 c_xr_pos_min[65]: - x[65] - x_pos[65] <= -150
 c_xr_pos_min[66]: - x[66] - x_pos[66] <= -150
 c_xr_pos_min[67]: - x[67] - x_pos[67] <= -150
 c_xr_pos_min[68]: - x[68] - x_pos[68] <= -150
 c_xr_pos_min[69]: - x[69] - x_pos[69] <= -150
 c_xr_pos_min[70]: - x[70] - x_pos[70] <= -150
 c_xr_pos_min[71]: - x[71] - x_pos[71] <= -150
 c_xr_pos_min[72]: - x[72] - x_pos[72] <= -150
 c_xr_pos_min[73]: - x[73] - x_pos[73] <= -150
 c_xr_pos_min[74]: - x[74] - x_pos[74] <= -150
 c_xr_pos_min[75]: - x[75] - x_pos[75] <= -150
 c_xr_pos_min[76]: - x[76] - x_pos[76] <= -150
 c_xr_pos_min[77]: - x[77] - x_pos[77] <= -150
 c_xr_pos_min[78]: - x[78] - x_pos[78] <= -150
 c_xr_pos_min[79]: - x[79] - x_pos[79] <= -150
 c_xr_pos_min[80]: - x[80] - x_pos[80] <= -150
 c_xr_pos_min[81]: - x[81] - x_pos[81] <= -150
 c_xr_pos_min[82]: - x[82] - x_pos[82] <= -150
 c_xr_pos_min[83]: - x[83] - x_pos[83] <= -150
 c_xr_pos_min[84]: - x[84] - x_pos[84] <= -150
 c_xr_pos_min[85]: - x[85] - x_pos[85] <= -150
 c_xr_pos_min[86]: - x[86] - x_pos[86] <= -150
 c_xr_pos_min[87]: - x[87] - x_pos[87] <= -150
 c_xr_pos_min[88]: - x[88] - x_pos[88] <= -150
 c_xr_pos_min[89]: - x[89] - x_pos[89] <= -150
 c_xr_pos_min[90]: - x[90] - x_pos[90] <= -150
 c_xr_pos_min[91]: - x[91] - x_pos[91] <= -150
 c_xr_pos_min[92]: - x[92] - x_pos[92] <= -150
 c_xr_pos_min[93]: - x[93] - x_pos[93] <= -150
 c_xr_pos_min[94]: - x[94] - x_pos[94] <= -150
 c_xr_pos_min[95]: - x[95] - x_pos[95] <= -150
 c_xr_pos_max[0]: x[0] + x_pos[0] <= 750
 c_xr_pos_max[1]: x[1] + x_pos[1] <= 750
 c_xr_pos_max[2]: x[2] + x_pos[2] <= 750
 c_xr_pos_max[3]: x[3] + x_pos[3] <= 750
 c_xr_pos_max[4]: x[4] + x_pos[4] <= 750
 c_xr_pos_max[5]: x[5] + x_pos[5] <= 750
 c_xr_pos_max[6]: x[6] + x_pos[6] <= 750
 c_xr_pos_max[7]: x[7] + x_pos[7] <= 750
 c_xr_pos_max[8]: x[8] + x_pos[8] <= 750
 c_xr_pos_max[9]: x[9] + x_pos[9] <= 750
 c_xr_pos_max[10]: x[10] + x_pos[10] <= 750
 c_xr_pos_max[11]: x[11] + x_pos[11] <= 750
 c_xr_pos_max[12]: x[12] + x_pos[12] <= 750
 c_xr_pos_max[13]: x[13] + x_pos[13] <= 750
 c_xr_pos_max[14]: x[14] + x_pos[14] <= 750
 c_xr_pos_max[15]: x[15] + x_pos[15] <= 750
 c_xr_pos_max[16]: x[16] + x_pos[16] <= 750
 c_xr_pos_max[17]: x[17] + x_pos[17] <= 750
 c_xr_pos_max[18]: x[18] + x_pos[18] <= 750
 c_xr_pos_max[19]: x[19] + x_pos[19] <= 750
 c_xr_pos_max[20]: x[20] + x_pos[20] <= 750
 c_xr_pos_max[21]: x[21] + x_pos[21] <= 750
 c_xr_pos_max[22]: x[22] + x_pos[22] <= 750
 c_xr_pos_max[23]: x[23] + x_pos[23] <= 750
 c_xr_pos_max[24]: x[24] + x_pos[24] <= 750
 c_xr_pos_max[25]: x[25] + x_pos[25] <= 750
 c_xr_pos_max[26]: x[26] + x_pos[26] <= 750
 c_xr_pos_max[27]: x[27] + x_pos[27] <= 750
 c_xr_pos_max[28]: x[28] + x_pos[28] <= 750
 c_xr_pos_max[29]: x[29] + x_pos[29] <= 750
 c_xr_pos_max[30]: x[30] + x_pos[30] <= 750
 c_xr_pos_max[31]: x[31] + x_pos[31] <= 750
 c_xr_pos_max[32]: x[32] + x_pos[32] <= 750
 c_xr_pos_max[33]: x[33] + x_pos[33] <= 750
 c_xr_pos_max[34]: x[34] + x_pos[34] <= 750
 c_xr_pos_max[35]: x[35] + x_pos[35] <= 750
 c_xr_pos_max[36]: x[36] + x_pos[36] <= 750
 c_xr_pos_max[37]: x[37] + x_pos[37] <= 750
 c_xr_pos_max[38]: x[38] + x_pos[38] <= 750
 c_xr_pos_max[39]: x[39] + x_pos[39] <= 750
 c_xr_pos_max[40]: x[40] + x_pos[40] <= 750
 c_xr_pos_max[41]: x[41] + x_pos[41] <= 750
 c_xr_pos_max[42]: x[42] + x_pos[42] <= 750
 c_xr_pos_max[43]: x[43] + x_pos[43] <= 750
 c_xr_pos_max[44]: x[44] + x_pos[44] <= 750
 c_xr_pos_max[45]: x[45] + x_pos[45] <= 750
 c_xr_pos_max[46]: x[46] + x_pos[46] <= 750
 c_xr_pos_max[47]: x[47] + x_pos[47] <= 750
 c_xr_pos_max[48]: x[48] + x_pos[48] <= 750
 c_xr_pos_max[49]: x[49] + x_pos[49] <= 750
 c_xr_pos_max[50]: x[50] + x_pos[50] <= 750
 c_xr_pos_max[51]: x[51] + x_pos[51] <= 750
 c_xr_pos_max[52]: x[52] + x_pos[52] <= 750
 c_xr_pos_max[53]: x[53] + x_pos[53] <= 750
 c_xr_pos_max[54]: x[54] + x_pos[54] <= 750
 c_xr_pos_max[55]: x[55] + x_pos[55] <= 750
 c_xr_pos_max[56]: x[56] + x_pos[56] <= 750
 c_xr_pos_max[57]: x[57] + x_pos[57] <= 750
 c_xr_pos_max[58]: x[58] + x_pos[58] <= 750
 c_xr_pos_max[59]: x[59] + x_pos[59] <= 750
 c_xr_pos_max[60]: x[60] + x_pos[60] <= 750
 c_xr_pos_max[61]: x[61] + x_pos[61] <= 750
 c_xr_pos_max[62]: x[62] + x_pos[62] <= 750
 c_xr_pos_max[63]: x[63] + x_pos[63] <= 750
 c_xr_pos_max[64]: x[64] + x_pos[64] <= 750
 c_xr_pos_max[65]: x[65] + x_pos[65] <= 750
 c_xr_pos_max[66]: x[66] + x_pos[66] <= 750
 c_xr_pos_max[67]: x[67] + x_pos[67] <= 750
 c_xr_pos_max[68]: x[68] + x_pos[68] <= 750
 c_xr_pos_max[69]: x[69] + x_pos[69] <= 750
 c_xr_pos_max[70]: x[70] + x_pos[70] <= 750
 c_xr_pos_max[71]: x[71] + x_pos[71] <= 750
 c_xr_pos_max[72]: x[72] + x_pos[72] <= 750
 c_xr_pos_max[73]: x[73] + x_pos[73] <= 750
 c_xr_pos_max[74]: x[74] + x_pos[74] <= 750
 c_xr_pos_max[75]: x[75] + x_pos[75] <= 750
 c_xr_pos_max[76]: x[76] + x_pos[76] <= 750
 c_xr_pos_max[77]: x[77] + x_pos[77] <= 750
 c_xr_pos_max[78]: x[78] + x_pos[78] <= 750
 c_xr_pos_max[79]: x[79] + x_pos[79] <= 750
 c_xr_pos_max[80]: x[80] + x_pos[80] <= 750
 c_xr_pos_max[81]: x[81] + x_pos[81] <= 750
 c_xr_pos_max[82]: x[82] + x_pos[82] <= 750
 c_xr_pos_max[83]: x[83] + x_pos[83] <= 750
 c_xr_pos_max[84]: x[84] + x_pos[84] <= 750
 c_xr_pos_max[85]: x[85] + x_pos[85] <= 750
 c_xr_pos_max[86]: x[86] + x_pos[86] <= 750
 c_xr_pos_max[87]: x[87] + x_pos[87] <= 750
 c_xr_pos_max[88]: x[88] + x_pos[88] <= 750
 c_xr_pos_max[89]: x[89] + x_pos[89] <= 750
 c_xr_pos_max[90]: x[90] + x_pos[90] <= 750
 c_xr_pos_max[91]: x[91] + x_pos[91] <= 750
 c_xr_pos_max[92]: x[92] + x_pos[92] <= 750
 c_xr_pos_max[93]: x[93] + x_pos[93] <= 750
 c_xr_pos_max[94]: x[94] + x_pos[94] <= 750
 c_xr_pos_max[95]: x[95] + x_pos[95] <= 750
 c_r_neg_min[0]: x_neg[0] <= 150
 c_r_neg_min[1]: x_neg[1] <= 150
 c_r_neg_min[2]: x_neg[2] <= 150
 c_r_neg_min[3]: x_neg[3] <= 150
 c_r_neg_min[4]: x_neg[4] <= 150
 c_r_neg_min[5]: x_neg[5] <= 150
 c_r_neg_min[6]: x_neg[6] <= 150
 c_r_neg_min[7]: x_neg[7] <= 150
 c_r_neg_min[8]: x_neg[8] <= 150
 c_r_neg_min[9]: x_neg[9] <= 150
 c_r_neg_min[10]: x_neg[10] <= 150
 c_r_neg_min[11]: x_neg[11] <= 150
 c_r_neg_min[12]: x_neg[12] <= 150
 c_r_neg_min[13]: x_neg[13] <= 150
 c_r_neg_min[14]: x_neg[14] <= 150
 c_r_neg_min[15]: x_neg[15] <= 150
 c_r_neg_min[16]: x_neg[16] <= 150
 c_r_neg_min[17]: x_neg[17] <= 150
 c_r_neg_min[18]: x_neg[18] <= 150
 c_r_neg_min[19]: x_neg[19] <= 150
 c_r_neg_min[20]: x_neg[20] <= 150
 c_r_neg_min[21]: x_neg[21] <= 150
 c_r_neg_min[22]: x_neg[22] <= 150
 c_r_neg_min[23]: x_neg[23] <= 150
 c_r_neg_min[24]: x_neg[24] <= 150
 c_r_neg_min[25]: x_neg[25] <= 150
 c_r_neg_min[26]: x_neg[26] <= 150
 c_r_neg_min[27]: x_neg[27] <= 150
 c_r_neg_min[28]: x_neg[28] <= 150
 c_r_neg_min[29]: x_neg[29] <= 150
 c_r_neg_min[30]: x_neg[30] <= 150
 c_r_neg_min[31]: x_neg[31] <= 150
 c_r_neg_min[32]: x_neg[32] <= 150
 c_r_neg_min[33]: x_neg[33] <= 150
 c_r_neg_min[34]: x_neg[34] <= 150
 c_r_neg_min[35]: x_neg[35] <= 150
 c_r_neg_min[36]: x_neg[36] <= 150
 c_r_neg_min[37]: x_neg[37] <= 150
 c_r_neg_min[38]: x_neg[38] <= 150
 c_r_neg_min[39]: x_neg[39] <= 150
 c_r_neg_min[40]: x_neg[40] <= 150
 c_r_neg_min[41]: x_neg[41] <= 150
 c_r_neg_min[42]: x_neg[42] <= 150
 c_r_neg_min[43]: x_neg[43] <= 150
 c_r_neg_min[44]: x_neg[44] <= 150
 c_r_neg_min[45]: x_neg[45] <= 150
 c_r_neg_min[46]: x_neg[46] <= 150
 c_r_neg_min[47]: x_neg[47] <= 150
 c_r_neg_min[48]: x_neg[48] <= 150
 c_r_neg_min[49]: x_neg[49] <= 150
 c_r_neg_min[50]: x_neg[50] <= 150
 c_r_neg_min[51]: x_neg[51] <= 150
 c_r_neg_min[52]: x_neg[52] <= 150
 c_r_neg_min[53]: x_neg[53] <= 150
 c_r_neg_min[54]: x_neg[54] <= 150
 c_r_neg_min[55]: x_neg[55] <= 150
 c_r_neg_min[56]: x_neg[56] <= 150
 c_r_neg_min[57]: x_neg[57] <= 150
 c_r_neg_min[58]: x_neg[58] <= 150
 c_r_neg_min[59]: x_neg[59] <= 150
 c_r_neg_min[60]: x_neg[60] <= 150
 c_r_neg_min[61]: x_neg[61] <= 150
 c_r_neg_min[62]: x_neg[62] <= 150
 c_r_neg_min[63]: x_neg[63] <= 150
 c_r_neg_min[64]: x_neg[64] <= 150
 c_r_neg_min[65]: x_neg[65] <= 150
 c_r_neg_min[66]: x_neg[66] <= 150
 c_r_neg_min[67]: x_neg[67] <= 150
 c_r_neg_min[68]: x_neg[68] <= 150
 c_r_neg_min[69]: x_neg[69] <= 150
 c_r_neg_min[70]: x_neg[70] <= 150
 c_r_neg_min[71]: x_neg[71] <= 150
 c_r_neg_min[72]: x_neg[72] <= 150
 c_r_neg_min[73]: x_neg[73] <= 150
 c_r_neg_min[74]: x_neg[74] <= 150
 c_r_neg_min[75]: x_neg[75] <= 150
 c_r_neg_min[76]: x_neg[76] <= 150
 c_r_neg_min[77]: x_neg[77] <= 150
 c_r_neg_min[78]: x_neg[78] <= 150
 c_r_neg_min[79]: x_neg[79] <= 150
 c_r_neg_min[80]: x_neg[80] <= 150
 c_r_neg_min[81]: x_neg[81] <= 150
 c_r_neg_min[82]: x_neg[82] <= 150
 c_r_neg_min[83]: x_neg[83] <= 150
 c_r_neg_min[84]: x_neg[84] <= 150
 c_r_neg_min[85]: x_neg[85] <= 150
 c_r_neg_min[86]: x_neg[86] <= 150
 c_r_neg_min[87]: x_neg[87] <= 150
 c_r_neg_min[88]: x_neg[88] <= 150
 c_r_neg_min[89]: x_neg[89] <= 150
 c_r_neg_min[90]: x_neg[90] <= 150
 c_r_neg_min[91]: x_neg[91] <= 150
 c_r_neg_min[92]: x_neg[92] <= 150
 c_r_neg_min[93]: x_neg[93] <= 150
 c_r_neg_min[94]: x_neg[94] <= 150
 c_r_neg_min[95]: x_neg[95] <= 150
 c_r_pos_min[0]: x_pos[0] <= 150
 c_r_pos_min[1]: x_pos[1] <= 150
 c_r_pos_min[2]: x_pos[2] <= 150
 c_r_pos_min[3]: x_pos[3] <= 150
 c_r_pos_min[4]: x_pos[4] <= 150
 c_r_pos_min[5]: x_pos[5] <= 150
 c_r_pos_min[6]: x_pos[6] <= 150
 c_r_pos_min[7]: x_pos[7] <= 150
 c_r_pos_min[8]: x_pos[8] <= 150
 c_r_pos_min[9]: x_pos[9] <= 150
 c_r_pos_min[10]: x_pos[10] <= 150
 c_r_pos_min[11]: x_pos[11] <= 150
 c_r_pos_min[12]: x_pos[12] <= 150
 c_r_pos_min[13]: x_pos[13] <= 150
 c_r_pos_min[14]: x_pos[14] <= 150
 c_r_pos_min[15]: x_pos[15] <= 150
 c_r_pos_min[16]: x_pos[16] <= 150
 c_r_pos_min[17]: x_pos[17] <= 150
 c_r_pos_min[18]: x_pos[18] <= 150
 c_r_pos_min[19]: x_pos[19] <= 150
 c_r_pos_min[20]: x_pos[20] <= 150
 c_r_pos_min[21]: x_pos[21] <= 150
 c_r_pos_min[22]: x_pos[22] <= 150
 c_r_pos_min[23]: x_pos[23] <= 150
 c_r_pos_min[24]: x_pos[24] <= 150
 c_r_pos_min[25]: x_pos[25] <= 150
 c_r_pos_min[26]: x_pos[26] <= 150
 c_r_pos_min[27]: x_pos[27] <= 150
 c_r_pos_min[28]: x_pos[28] <= 150
 c_r_pos_min[29]: x_pos[29] <= 150
 c_r_pos_min[30]: x_pos[30] <= 150
 c_r_pos_min[31]: x_pos[31] <= 150
 c_r_pos_min[32]: x_pos[32] <= 150
 c_r_pos_min[33]: x_pos[33] <= 150
 c_r_pos_min[34]: x_pos[34] <= 150
 c_r_pos_min[35]: x_pos[35] <= 150
 c_r_pos_min[36]: x_pos[36] <= 150
 c_r_pos_min[37]: x_pos[37] <= 150
 c_r_pos_min[38]: x_pos[38] <= 150
 c_r_pos_min[39]: x_pos[39] <= 150
 c_r_pos_min[40]: x_pos[40] <= 150
 c_r_pos_min[41]: x_pos[41] <= 150
 c_r_pos_min[42]: x_pos[42] <= 150
 c_r_pos_min[43]: x_pos[43] <= 150
 c_r_pos_min[44]: x_pos[44] <= 150
 c_r_pos_min[45]: x_pos[45] <= 150
 c_r_pos_min[46]: x_pos[46] <= 150
 c_r_pos_min[47]: x_pos[47] <= 150
 c_r_pos_min[48]: x_pos[48] <= 150
 c_r_pos_min[49]: x_pos[49] <= 150
 c_r_pos_min[50]: x_pos[50] <= 150
 c_r_pos_min[51]: x_pos[51] <= 150
 c_r_pos_min[52]: x_pos[52] <= 150
 c_r_pos_min[53]: x_pos[53] <= 150
 c_r_pos_min[54]: x_pos[54] <= 150
 c_r_pos_min[55]: x_pos[55] <= 150
 c_r_pos_min[56]: x_pos[56] <= 150
 c_r_pos_min[57]: x_pos[57] <= 150
 c_r_pos_min[58]: x_pos[58] <= 150
 c_r_pos_min[59]: x_pos[59] <= 150
 c_r_pos_min[60]: x_pos[60] <= 150
 c_r_pos_min[61]: x_pos[61] <= 150
 c_r_pos_min[62]: x_pos[62] <= 150
 c_r_pos_min[63]: x_pos[63] <= 150
 c_r_pos_min[64]: x_pos[64] <= 150
 c_r_pos_min[65]: x_pos[65] <= 150
 c_r_pos_min[66]: x_pos[66] <= 150
 c_r_pos_min[67]: x_pos[67] <= 150
 c_r_pos_min[68]: x_pos[68] <= 150
 c_r_pos_min[69]: x_pos[69] <= 150
 c_r_pos_min[70]: x_pos[70] <= 150
 c_r_pos_min[71]: x_pos[71] <= 150
 c_r_pos_min[72]: x_pos[72] <= 150
 c_r_pos_min[73]: x_pos[73] <= 150
 c_r_pos_min[74]: x_pos[74] <= 150
 c_r_pos_min[75]: x_pos[75] <= 150
 c_r_pos_min[76]: x_pos[76] <= 150
 c_r_pos_min[77]: x_pos[77] <= 150
 c_r_pos_min[78]: x_pos[78] <= 150
 c_r_pos_min[79]: x_pos[79] <= 150
 c_r_pos_min[80]: x_pos[80] <= 150
 c_r_pos_min[81]: x_pos[81] <= 150
 c_r_pos_min[82]: x_pos[82] <= 150
 c_r_pos_min[83]: x_pos[83] <= 150
 c_r_pos_min[84]: x_pos[84] <= 150
 c_r_pos_min[85]: x_pos[85] <= 150
 c_r_pos_min[86]: x_pos[86] <= 150
 c_r_pos_min[87]: x_pos[87] <= 150
 c_r_pos_min[88]: x_pos[88] <= 150
 c_r_pos_min[89]: x_pos[89] <= 150
 c_r_pos_min[90]: x_pos[90] <= 150
 c_r_pos_min[91]: x_pos[91] <= 150
 c_r_pos_min[92]: x_pos[92] <= 150
 c_r_pos_min[93]: x_pos[93] <= 150
 c_r_pos_min[94]: x_pos[94] <= 150
 c_r_pos_min[95]: x_pos[95] <= 150
 c_chgarge_max[0]: - 250 x_b[0] + x_chg[0] <= 0
 c_chgarge_max[1]: - 250 x_b[1] + x_chg[1] <= 0
 c_chgarge_max[2]: - 250 x_b[2] + x_chg[2] <= 0
 c_chgarge_max[3]: - 250 x_b[3] + x_chg[3] <= 0
 c_chgarge_max[4]: - 250 x_b[4] + x_chg[4] <= 0
 c_chgarge_max[5]: - 250 x_b[5] + x_chg[5] <= 0
 c_chgarge_max[6]: - 250 x_b[6] + x_chg[6] <= 0
 c_chgarge_max[7]: - 250 x_b[7] + x_chg[7] <= 0
 c_chgarge_max[8]: - 250 x_b[8] + x_chg[8] <= 0
 c_chgarge_max[9]: - 250 x_b[9] + x_chg[9] <= 0
 c_chgarge_max[10]: - 250 x_b[10] + x_chg[10] <= 0
 c_chgarge_max[11]: - 250 x_b[11] + x_chg[11] <= 0
 c_chgarge_max[12]: - 250 x_b[12] + x_chg[12] <= 0
 c_chgarge_max[13]: - 250 x_b[13] + x_chg[13] <= 0
 c_chgarge_max[14]: - 250 x_b[14] + x_chg[14] <= 0
 c_chgarge_max[15]: - 250 x_b[15] + x_chg[15] <= 0
 c_chgarge_max[16]: - 250 x_b[16] + x_chg[16] <= 0
 c_chgarge_max[17]: - 250 x_b[17] + x_chg[17] <= 0
 c_chgarge_max[18]: - 250 x_b[18] + x_chg[18] <= 0
 c_chgarge_max[19]: - 250 x_b[19] + x_chg[19] <= 0
 c_chgarge_max[20]: - 250 x_b[20] + x_chg[20] <= 0
 c_chgarge_max[21]: - 250 x_b[21] + x_chg[21] <= 0
 c_chgarge_max[22]: - 250 x_b[22] + x_chg[22] <= 0
 c_chgarge_max[23]: - 250 x_b[23] + x_chg[23] <= 0
 c_chgarge_max[24]: - 250 x_b[24] + x_chg[24] <= 0
 c_chgarge_max[25]: - 250 x_b[25] + x_chg[25] <= 0
 c_chgarge_max[26]: - 250 x_b[26] + x_chg[26] <= 0
 c_chgarge_max[27]: - 250 x_b[27] + x_chg[27] <= 0
 c_chgarge_max[28]: - 250 x_b[28] + x_chg[28] <= 0
 c_chgarge_max[29]: - 250 x_b[29] + x_chg[29] <= 0
 c_chgarge_max[30]: - 250 x_b[30] + x_chg[30] <= 0
 c_chgarge_max[31]: - 250 x_b[31] + x_chg[31] <= 0
 c_chgarge_max[32]: - 250 x_b[32] + x_chg[32] <= 0
 c_chgarge_max[33]: - 250 x_b[33] + x_chg[33] <= 0
 c_chgarge_max[34]: - 250 x_b[34] + x_chg[34] <= 0
 c_chgarge_max[35]: - 250 x_b[35] + x_chg[35] <= 0
 c_chgarge_max[36]: - 250 x_b[36] + x_chg[36] <= 0
 c_chgarge_max[37]: - 250 x_b[37] + x_chg[37] <= 0
 c_chgarge_max[38]: - 250 x_b[38] + x_chg[38] <= 0
 c_chgarge_max[39]: - 250 x_b[39] + x_chg[39] <= 0
 c_chgarge_max[40]: - 250 x_b[40] + x_chg[40] <= 0
 c_chgarge_max[41]: - 250 x_b[41] + x_chg[41] <= 0
 c_chgarge_max[42]: - 250 x_b[42] + x_chg[42] <= 0
 c_chgarge_max[43]: - 250 x_b[43] + x_chg[43] <= 0
 c_chgarge_max[44]: - 250 x_b[44] + x_chg[44] <= 0
 c_chgarge_max[45]: - 250 x_b[45] + x_chg[45] <= 0
 c_chgarge_max[46]: - 250 x_b[46] + x_chg[46] <= 0
 c_chgarge_max[47]: - 250 x_b[47] + x_chg[47] <= 0
 c_chgarge_max[48]: - 250 x_b[48] + x_chg[48] <= 0
 c_chgarge_max[49]: - 250 x_b[49] + x_chg[49] <= 0
 c_chgarge_max[50]: - 250 x_b[50] + x_chg[50] <= 0
 c_chgarge_max[51]: - 250 x_b[51] + x_chg[51] <= 0
 c_chgarge_max[52]: - 250 x_b[52] + x_chg[52] <= 0
 c_chgarge_max[53]: - 250 x_b[53] + x_chg[53] <= 0
 c_chgarge_max[54]: - 250 x_b[54] + x_chg[54] <= 0
 c_chgarge_max[55]: - 250 x_b[55] + x_chg[55] <= 0
 c_chgarge_max[56]: - 250 x_b[56] + x_chg[56] <= 0
 c_chgarge_max[57]: - 250 x_b[57] + x_chg[57] <= 0
 c_chgarge_max[58]: - 250 x_b[58] + x_chg[58] <= 0
 c_chgarge_max[59]: - 250 x_b[59] + x_chg[59] <= 0
 c_chgarge_max[60]: - 250 x_b[60] + x_chg[60] <= 0
 c_chgarge_max[61]: - 250 x_b[61] + x_chg[61] <= 0
 c_chgarge_max[62]: - 250 x_b[62] + x_chg[62] <= 0
 c_chgarge_max[63]: - 250 x_b[63] + x_chg[63] <= 0
 c_chgarge_max[64]: - 250 x_b[64] + x_chg[64] <= 0
 c_chgarge_max[65]: - 250 x_b[65] + x_chg[65] <= 0
 c_chgarge_max[66]: - 250 x_b[66] + x_chg[66] <= 0
 c_chgarge_max[67]: - 250 x_b[67] + x_chg[67] <= 0
 c_chgarge_max[68]: - 250 x_b[68] + x_chg[68] <= 0
 c_chgarge_max[69]: - 250 x_b[69] + x_chg[69] <= 0
 c_chgarge_max[70]: - 250 x_b[70] + x_chg[70] <= 0
 c_chgarge_max[71]: - 250 x_b[71] + x_chg[71] <= 0
 c_chgarge_max[72]: - 250 x_b[72] + x_chg[72] <= 0
 c_chgarge_max[73]: - 250 x_b[73] + x_chg[73] <= 0
 c_chgarge_max[74]: - 250 x_b[74] + x_chg[74] <= 0
 c_chgarge_max[75]: - 250 x_b[75] + x_chg[75] <= 0
 c_chgarge_max[76]: - 250 x_b[76] + x_chg[76] <= 0
 c_chgarge_max[77]: - 250 x_b[77] + x_chg[77] <= 0
 c_chgarge_max[78]: - 250 x_b[78] + x_chg[78] <= 0
 c_chgarge_max[79]: - 250 x_b[79] + x_chg[79] <= 0
 c_chgarge_max[80]: - 250 x_b[80] + x_chg[80] <= 0
 c_chgarge_max[81]: - 250 x_b[81] + x_chg[81] <= 0
 c_chgarge_max[82]: - 250 x_b[82] + x_chg[82] <= 0
 c_chgarge_max[83]: - 250 x_b[83] + x_chg[83] <= 0
 c_chgarge_max[84]: - 250 x_b[84] + x_chg[84] <= 0
 c_chgarge_max[85]: - 250 x_b[85] + x_chg[85] <= 0
 c_chgarge_max[86]: - 250 x_b[86] + x_chg[86] <= 0
 c_chgarge_max[87]: - 250 x_b[87] + x_chg[87] <= 0
 c_chgarge_max[88]: - 250 x_b[88] + x_chg[88] <= 0
 c_chgarge_max[89]: - 250 x_b[89] + x_chg[89] <= 0
 c_chgarge_max[90]: - 250 x_b[90] + x_chg[90] <= 0
 c_chgarge_max[91]: - 250 x_b[91] + x_chg[91] <= 0
 c_chgarge_max[92]: - 250 x_b[92] + x_chg[92] <= 0
 c_chgarge_max[93]: - 250 x_b[93] + x_chg[93] <= 0
 c_chgarge_max[94]: - 250 x_b[94] + x_chg[94] <= 0
 c_chgarge_max[95]: - 250 x_b[95] + x_chg[95] <= 0
 c_discharge_max[0]: 250 x_b[0] + x_dis[0] <= 250
 c_discharge_max[1]: 250 x_b[1] + x_dis[1] <= 250
 c_discharge_max[2]: 250 x_b[2] + x_dis[2] <= 250
 c_discharge_max[3]: 250 x_b[3] + x_dis[3] <= 250
 c_discharge_max[4]: 250 x_b[4] + x_dis[4] <= 250
 c_discharge_max[5]: 250 x_b[5] + x_dis[5] <= 250
 c_discharge_max[6]: 250 x_b[6] + x_dis[6] <= 250
 c_discharge_max[7]: 250 x_b[7] + x_dis[7] <= 250
 c_discharge_max[8]: 250 x_b[8] + x_dis[8] <= 250
 c_discharge_max[9]: 250 x_b[9] + x_dis[9] <= 250
 c_discharge_max[10]: 250 x_b[10] + x_dis[10] <= 250
 c_discharge_max[11]: 250 x_b[11] + x_dis[11] <= 250
 c_discharge_max[12]: 250 x_b[12] + x_dis[12] <= 250
 c_discharge_max[13]: 250 x_b[13] + x_dis[13] <= 250
 c_discharge_max[14]: 250 x_b[14] + x_dis[14] <= 250
 c_discharge_max[15]: 250 x_b[15] + x_dis[15] <= 250
 c_discharge_max[16]: 250 x_b[16] + x_dis[16] <= 250
 c_discharge_max[17]: 250 x_b[17] + x_dis[17] <= 250
 c_discharge_max[18]: 250 x_b[18] + x_dis[18] <= 250
 c_discharge_max[19]: 250 x_b[19] + x_dis[19] <= 250
 c_discharge_max[20]: 250 x_b[20] + x_dis[20] <= 250
 c_discharge_max[21]: 250 x_b[21] + x_dis[21] <= 250
 c_discharge_max[22]: 250 x_b[22] + x_dis[22] <= 250
 c_discharge_max[23]: 250 x_b[23] + x_dis[23] <= 250
 c_discharge_max[24]: 250 x_b[24] + x_dis[24] <= 250
 c_discharge_max[25]: 250 x_b[25] + x_dis[25] <= 250
 c_discharge_max[26]: 250 x_b[26] + x_dis[26] <= 250
 c_discharge_max[27]: 250 x_b[27] + x_dis[27] <= 250
 c_discharge_max[28]: 250 x_b[28] + x_dis[28] <= 250
 c_discharge_max[29]: 250 x_b[29] + x_dis[29] <= 250
 c_discharge_max[30]: 250 x_b[30] + x_dis[30] <= 250
 c_discharge_max[31]: 250 x_b[31] + x_dis[31] <= 250
 c_discharge_max[32]: 250 x_b[32] + x_dis[32] <= 250
 c_discharge_max[33]: 250 x_b[33] + x_dis[33] <= 250
 c_discharge_max[34]: 250 x_b[34] + x_dis[34] <= 250
 c_discharge_max[35]: 250 x_b[35] + x_dis[35] <= 250
 c_discharge_max[36]: 250 x_b[36] + x_dis[36] <= 250
 c_discharge_max[37]: 250 x_b[37] + x_dis[37] <= 250
 c_discharge_max[38]: 250 x_b[38] + x_dis[38] <= 250
 c_discharge_max[39]: 250 x_b[39] + x_dis[39] <= 250
 c_discharge_max[40]: 250 x_b[40] + x_dis[40] <= 250
 c_discharge_max[41]: 250 x_b[41] + x_dis[41] <= 250
 c_discharge_max[42]: 250 x_b[42] + x_dis[42] <= 250
 c_discharge_max[43]: 250 x_b[43] + x_dis[43] <= 250
 c_discharge_max[44]: 250 x_b[44] + x_dis[44] <= 250
 c_discharge_max[45]: 250 x_b[45] + x_dis[45] <= 250
 c_discharge_max[46]: 250 x_b[46] + x_dis[46] <= 250
 c_discharge_max[47]: 250 x_b[47] + x_dis[47] <= 250
 c_discharge_max[48]: 250 x_b[48] + x_dis[48] <= 250
 c_discharge_max[49]: 250 x_b[49] + x_dis[49] <= 250
 c_discharge_max[50]: 250 x_b[50] + x_dis[50] <= 250
 c_discharge_max[51]: 250 x_b[51] + x_dis[51] <= 250
 c_discharge_max[52]: 250 x_b[52] + x_dis[52] <= 250
 c_discharge_max[53]: 250 x_b[53] + x_dis[53] <= 250
 c_discharge_max[54]: 250 x_b[54] + x_dis[54] <= 250
 c_discharge_max[55]: 250 x_b[55] + x_dis[55] <= 250
 c_discharge_max[56]: 250 x_b[56] + x_dis[56] <= 250
 c_discharge_max[57]: 250 x_b[57] + x_dis[57] <= 250
 c_discharge_max[58]: 250 x_b[58] + x_dis[58] <= 250
 c_discharge_max[59]: 250 x_b[59] + x_dis[59] <= 250
 c_discharge_max[60]: 250 x_b[60] + x_dis[60] <= 250
 c_discharge_max[61]: 250 x_b[61] + x_dis[61] <= 250
 c_discharge_max[62]: 250 x_b[62] + x_dis[62] <= 250
 c_discharge_max[63]: 250 x_b[63] + x_dis[63] <= 250
 c_discharge_max[64]: 250 x_b[64] + x_dis[64] <= 250
 c_discharge_max[65]: 250 x_b[65] + x_dis[65] <= 250
 c_discharge_max[66]: 250 x_b[66] + x_dis[66] <= 250
 c_discharge_max[67]: 250 x_b[67] + x_dis[67] <= 250
 c_discharge_max[68]: 250 x_b[68] + x_dis[68] <= 250
 c_discharge_max[69]: 250 x_b[69] + x_dis[69] <= 250
 c_discharge_max[70]: 250 x_b[70] + x_dis[70] <= 250
 c_discharge_max[71]: 250 x_b[71] + x_dis[71] <= 250
 c_discharge_max[72]: 250 x_b[72] + x_dis[72] <= 250
 c_discharge_max[73]: 250 x_b[73] + x_dis[73] <= 250
 c_discharge_max[74]: 250 x_b[74] + x_dis[74] <= 250
 c_discharge_max[75]: 250 x_b[75] + x_dis[75] <= 250
 c_discharge_max[76]: 250 x_b[76] + x_dis[76] <= 250
 c_discharge_max[77]: 250 x_b[77] + x_dis[77] <= 250
 c_discharge_max[78]: 250 x_b[78] + x_dis[78] <= 250
 c_discharge_max[79]: 250 x_b[79] + x_dis[79] <= 250
 c_discharge_max[80]: 250 x_b[80] + x_dis[80] <= 250
 c_discharge_max[81]: 250 x_b[81] + x_dis[81] <= 250
 c_discharge_max[82]: 250 x_b[82] + x_dis[82] <= 250
 c_discharge_max[83]: 250 x_b[83] + x_dis[83] <= 250
 c_discharge_max[84]: 250 x_b[84] + x_dis[84] <= 250
 c_discharge_max[85]: 250 x_b[85] + x_dis[85] <= 250
 c_discharge_max[86]: 250 x_b[86] + x_dis[86] <= 250
 c_discharge_max[87]: 250 x_b[87] + x_dis[87] <= 250
 c_discharge_max[88]: 250 x_b[88] + x_dis[88] <= 250
 c_discharge_max[89]: 250 x_b[89] + x_dis[89] <= 250
 c_discharge_max[90]: 250 x_b[90] + x_dis[90] <= 250
 c_discharge_max[91]: 250 x_b[91] + x_dis[91] <= 250
 c_discharge_max[92]: 250 x_b[92] + x_dis[92] <= 250
 c_discharge_max[93]: 250 x_b[93] + x_dis[93] <= 250
 c_discharge_max[94]: 250 x_b[94] + x_dis[94] <= 250
 c_discharge_max[95]: 250 x_b[95] + x_dis[95] <= 250
 c_SOC_min[0]: - x_S[0] <= -170.1
 c_SOC_min[1]: - x_S[1] <= -170.1
 c_SOC_min[2]: - x_S[2] <= -170.1
 c_SOC_min[3]: - x_S[3] <= -170.1
 c_SOC_min[4]: - x_S[4] <= -170.1
 c_SOC_min[5]: - x_S[5] <= -170.1
 c_SOC_min[6]: - x_S[6] <= -170.1
 c_SOC_min[7]: - x_S[7] <= -170.1
 c_SOC_min[8]: - x_S[8] <= -170.1
 c_SOC_min[9]: - x_S[9] <= -170.1
 c_SOC_min[10]: - x_S[10] <= -170.1
 c_SOC_min[11]: - x_S[11] <= -170.1
 c_SOC_min[12]: - x_S[12] <= -170.1
 c_SOC_min[13]: - x_S[13] <= -170.1
 c_SOC_min[14]: - x_S[14] <= -170.1
 c_SOC_min[15]: - x_S[15] <= -170.1
 c_SOC_min[16]: - x_S[16] <= -170.1
 c_SOC_min[17]: - x_S[17] <= -170.1
 c_SOC_min[18]: - x_S[18] <= -170.1
 c_SOC_min[19]: - x_S[19] <= -170.1
 c_SOC_min[20]: - x_S[20] <= -170.1
 c_SOC_min[21]: - x_S[21] <= -170.1
 c_SOC_min[22]: - x_S[22] <= -170.1
 c_SOC_min[23]: - x_S[23] <= -170.1
 c_SOC_min[24]: - x_S[24] <= -170.1
 c_SOC_min[25]: - x_S[25] <= -170.1
 c_SOC_min[26]: - x_S[26] <= -170.1
 c_SOC_min[27]: - x_S[27] <= -170.1
 c_SOC_min[28]: - x_S[28] <= -170.1
 c_SOC_min[29]: - x_S[29] <= -170.1
 c_SOC_min[30]: - x_S[30] <= -170.1
 c_SOC_min[31]: - x_S[31] <= -170.1
 c_SOC_min[32]: - x_S[32] <= -170.1
 c_SOC_min[33]: - x_S[33] <= -170.1
 c_SOC_min[34]: - x_S[34] <= -170.1
 c_SOC_min[35]: - x_S[35] <= -170.1
 c_SOC_min[36]: - x_S[36] <= -170.1
 c_SOC_min[37]: - x_S[37] <= -170.1
 c_SOC_min[38]: - x_S[38] <= -170.1
 c_SOC_min[39]: - x_S[39] <= -170.1
 c_SOC_min[40]: - x_S[40] <= -170.1
 c_SOC_min[41]: - x_S[41] <= -170.1
 c_SOC_min[42]: - x_S[42] <= -170.1
 c_SOC_min[43]: - x_S[43] <= -170.1
 c_SOC_min[44]: - x_S[44] <= -170.1
 c_SOC_min[45]: - x_S[45] <= -170.1
 c_SOC_min[46]: - x_S[46] <= -170.1
 c_SOC_min[47]: - x_S[47] <= -170.1
 c_SOC_min[48]: - x_S[48] <= -170.1
 c_SOC_min[49]: - x_S[49] <= -170.1
 c_SOC_min[50]: - x_S[50] <= -170.1
 c_SOC_min[51]: - x_S[51] <= -170.1
 c_SOC_min[52]: - x_S[52] <= -170.1
 c_SOC_min[53]: - x_S[53] <= -170.1
 c_SOC_min[54]: - x_S[54] <= -170.1
 c_SOC_min[55]: - x_S[55] <= -170.1
 c_SOC_min[56]: - x_S[56] <= -170.1
 c_SOC_min[57]: - x_S[57] <= -170.1
 c_SOC_min[58]: - x_S[58] <= -170.1
 c_SOC_min[59]: - x_S[59] <= -170.1
 c_SOC_min[60]: - x_S[60] <= -170.1
 c_SOC_min[61]: - x_S[61] <= -170.1
 c_SOC_min[62]: - x_S[62] <= -170.1
 c_SOC_min[63]: - x_S[63] <= -170.1
 c_SOC_min[64]: - x_S[64] <= -170.1
 c_SOC_min[65]: - x_S[65] <= -170.1
 c_SOC_min[66]: - x_S[66] <= -170.1
 c_SOC_min[67]: - x_S[67] <= -170.1
 c_SOC_min[68]: - x_S[68] <= -170.1
 c_SOC_min[69]: - x_S[69] <= -170.1
 c_SOC_min[70]: - x_S[70] <= -170.1
 c_SOC_min[71]: - x_S[71] <= -170.1
 c_SOC_min[72]: - x_S[72] <= -170.1
 c_SOC_min[73]: - x_S[73] <= -170.1
 c_SOC_min[74]: - x_S[74] <= -170.1
 c_SOC_min[75]: - x_S[75] <= -170.1
 c_SOC_min[76]: - x_S[76] <= -170.1
 c_SOC_min[77]: - x_S[77] <= -170.1
 c_SOC_min[78]: - x_S[78] <= -170.1
 c_SOC_min[79]: - x_S[79] <= -170.1
 c_SOC_min[80]: - x_S[80] <= -170.1
 c_SOC_min[81]: - x_S[81] <= -170.1
 c_SOC_min[82]: - x_S[82] <= -170.1
 c_SOC_min[83]: - x_S[83] <= -170.1
 c_SOC_min[84]: - x_S[84] <= -170.1
 c_SOC_min[85]: - x_S[85] <= -170.1
 c_SOC_min[86]: - x_S[86] <= -170.1
 c_SOC_min[87]: - x_S[87] <= -170.1
 c_SOC_min[88]: - x_S[88] <= -170.1
 c_SOC_min[89]: - x_S[89] <= -170.1
 c_SOC_min[90]: - x_S[90] <= -170.1
 c_SOC_min[91]: - x_S[91] <= -170.1
 c_SOC_min[92]: - x_S[92] <= -170.1
 c_SOC_min[93]: - x_S[93] <= -170.1
 c_SOC_min[94]: - x_S[94] <= -170.1
 c_SOC_min[95]: - x_S[95] <= -170.1
 c_SOC_max[0]: x_S[0] <= 396.9
 c_SOC_max[1]: x_S[1] <= 396.9
 c_SOC_max[2]: x_S[2] <= 396.9
 c_SOC_max[3]: x_S[3] <= 396.9
 c_SOC_max[4]: x_S[4] <= 396.9
 c_SOC_max[5]: x_S[5] <= 396.9
 c_SOC_max[6]: x_S[6] <= 396.9
 c_SOC_max[7]: x_S[7] <= 396.9
 c_SOC_max[8]: x_S[8] <= 396.9
 c_SOC_max[9]: x_S[9] <= 396.9
 c_SOC_max[10]: x_S[10] <= 396.9
 c_SOC_max[11]: x_S[11] <= 396.9
 c_SOC_max[12]: x_S[12] <= 396.9
 c_SOC_max[13]: x_S[13] <= 396.9
 c_SOC_max[14]: x_S[14] <= 396.9
 c_SOC_max[15]: x_S[15] <= 396.9
 c_SOC_max[16]: x_S[16] <= 396.9
 c_SOC_max[17]: x_S[17] <= 396.9
 c_SOC_max[18]: x_S[18] <= 396.9
 c_SOC_max[19]: x_S[19] <= 396.9
 c_SOC_max[20]: x_S[20] <= 396.9
 c_SOC_max[21]: x_S[21] <= 396.9
 c_SOC_max[22]: x_S[22] <= 396.9
 c_SOC_max[23]: x_S[23] <= 396.9
 c_SOC_max[24]: x_S[24] <= 396.9
 c_SOC_max[25]: x_S[25] <= 396.9
 c_SOC_max[26]: x_S[26] <= 396.9
 c_SOC_max[27]: x_S[27] <= 396.9
 c_SOC_max[28]: x_S[28] <= 396.9
 c_SOC_max[29]: x_S[29] <= 396.9
 c_SOC_max[30]: x_S[30] <= 396.9
 c_SOC_max[31]: x_S[31] <= 396.9
 c_SOC_max[32]: x_S[32] <= 396.9
 c_SOC_max[33]: x_S[33] <= 396.9
 c_SOC_max[34]: x_S[34] <= 396.9
 c_SOC_max[35]: x_S[35] <= 396.9
 c_SOC_max[36]: x_S[36] <= 396.9
 c_SOC_max[37]: x_S[37] <= 396.9
 c_SOC_max[38]: x_S[38] <= 396.9
 c_SOC_max[39]: x_S[39] <= 396.9
 c_SOC_max[40]: x_S[40] <= 396.9
 c_SOC_max[41]: x_S[41] <= 396.9
 c_SOC_max[42]: x_S[42] <= 396.9
 c_SOC_max[43]: x_S[43] <= 396.9
 c_SOC_max[44]: x_S[44] <= 396.9
 c_SOC_max[45]: x_S[45] <= 396.9
 c_SOC_max[46]: x_S[46] <= 396.9
 c_SOC_max[47]: x_S[47] <= 396.9
 c_SOC_max[48]: x_S[48] <= 396.9
 c_SOC_max[49]: x_S[49] <= 396.9
 c_SOC_max[50]: x_S[50] <= 396.9
 c_SOC_max[51]: x_S[51] <= 396.9
 c_SOC_max[52]: x_S[52] <= 396.9
 c_SOC_max[53]: x_S[53] <= 396.9
 c_SOC_max[54]: x_S[54] <= 396.9
 c_SOC_max[55]: x_S[55] <= 396.9
 c_SOC_max[56]: x_S[56] <= 396.9
 c_SOC_max[57]: x_S[57] <= 396.9
 c_SOC_max[58]: x_S[58] <= 396.9
 c_SOC_max[59]: x_S[59] <= 396.9
 c_SOC_max[60]: x_S[60] <= 396.9
 c_SOC_max[61]: x_S[61] <= 396.9
 c_SOC_max[62]: x_S[62] <= 396.9
 c_SOC_max[63]: x_S[63] <= 396.9
 c_SOC_max[64]: x_S[64] <= 396.9
 c_SOC_max[65]: x_S[65] <= 396.9
 c_SOC_max[66]: x_S[66] <= 396.9
 c_SOC_max[67]: x_S[67] <= 396.9
 c_SOC_max[68]: x_S[68] <= 396.9
 c_SOC_max[69]: x_S[69] <= 396.9
 c_SOC_max[70]: x_S[70] <= 396.9
 c_SOC_max[71]: x_S[71] <= 396.9
 c_SOC_max[72]: x_S[72] <= 396.9
 c_SOC_max[73]: x_S[73] <= 396.9
 c_SOC_max[74]: x_S[74] <= 396.9
 c_SOC_max[75]: x_S[75] <= 396.9
 c_SOC_max[76]: x_S[76] <= 396.9
 c_SOC_max[77]: x_S[77] <= 396.9
 c_SOC_max[78]: x_S[78] <= 396.9
 c_SOC_max[79]: x_S[79] <= 396.9
 c_SOC_max[80]: x_S[80] <= 396.9
 c_SOC_max[81]: x_S[81] <= 396.9
 c_SOC_max[82]: x_S[82] <= 396.9
 c_SOC_max[83]: x_S[83] <= 396.9
 c_SOC_max[84]: x_S[84] <= 396.9
 c_SOC_max[85]: x_S[85] <= 396.9
 c_SOC_max[86]: x_S[86] <= 396.9
 c_SOC_max[87]: x_S[87] <= 396.9
 c_SOC_max[88]: x_S[88] <= 396.9
 c_SOC_max[89]: x_S[89] <= 396.9
 c_SOC_max[90]: x_S[90] <= 396.9
 c_SOC_max[91]: x_S[91] <= 396.9
 c_SOC_max[92]: x_S[92] <= 396.9
 c_SOC_max[93]: x_S[93] <= 396.9
 c_SOC_max[94]: x_S[94] <= 396.9
 c_SOC_max[95]: x_S[95] <= 396.9
 c_SOC_first: - 0.2325 x_chg[0] + 0.2688172043010753 x_dis[0] + x_S[0]
   = 187.5
 c_SOC_dynamic[1]: - 0.2325 x_chg[1] + 0.2688172043010753 x_dis[1] - x_S[0]
   + x_S[1] = 0
 c_SOC_dynamic[2]: - 0.2325 x_chg[2] + 0.2688172043010753 x_dis[2] - x_S[1]
   + x_S[2] = 0
 c_SOC_dynamic[3]: - 0.2325 x_chg[3] + 0.2688172043010753 x_dis[3] - x_S[2]
   + x_S[3] = 0
 c_SOC_dynamic[4]: - 0.2325 x_chg[4] + 0.2688172043010753 x_dis[4] - x_S[3]
   + x_S[4] = 0
 c_SOC_dynamic[5]: - 0.2325 x_chg[5] + 0.2688172043010753 x_dis[5] - x_S[4]
   + x_S[5] = 0
 c_SOC_dynamic[6]: - 0.2325 x_chg[6] + 0.2688172043010753 x_dis[6] - x_S[5]
   + x_S[6] = 0
 c_SOC_dynamic[7]: - 0.2325 x_chg[7] + 0.2688172043010753 x_dis[7] - x_S[6]
   + x_S[7] = 0
 c_SOC_dynamic[8]: - 0.2325 x_chg[8] + 0.2688172043010753 x_dis[8] - x_S[7]
   + x_S[8] = 0
 c_SOC_dynamic[9]: - 0.2325 x_chg[9] + 0.2688172043010753 x_dis[9] - x_S[8]
   + x_S[9] = 0
 c_SOC_dynamic[10]: - 0.2325 x_chg[10] + 0.2688172043010753 x_dis[10]
   - x_S[9] + x_S[10] = 0
 c_SOC_dynamic[11]: - 0.2325 x_chg[11] + 0.2688172043010753 x_dis[11]
   - x_S[10] + x_S[11] = 0
 c_SOC_dynamic[12]: - 0.2325 x_chg[12] + 0.2688172043010753 x_dis[12]
   - x_S[11] + x_S[12] = 0
 c_SOC_dynamic[13]: - 0.2325 x_chg[13] + 0.2688172043010753 x_dis[13]
   - x_S[12] + x_S[13] = 0
 c_SOC_dynamic[14]: - 0.2325 x_chg[14] + 0.2688172043010753 x_dis[14]
   - x_S[13] + x_S[14] = 0
 c_SOC_dynamic[15]: - 0.2325 x_chg[15] + 0.2688172043010753 x_dis[15]
   - x_S[14] + x_S[15] = 0
 c_SOC_dynamic[16]: - 0.2325 x_chg[16] + 0.2688172043010753 x_dis[16]
   - x_S[15] + x_S[16] = 0
 c_SOC_dynamic[17]: - 0.2325 x_chg[17] + 0.2688172043010753 x_dis[17]
   - x_S[16] + x_S[17] = 0
 c_SOC_dynamic[18]: - 0.2325 x_chg[18] + 0.2688172043010753 x_dis[18]
   - x_S[17] + x_S[18] = 0
 c_SOC_dynamic[19]: - 0.2325 x_chg[19] + 0.2688172043010753 x_dis[19]
   - x_S[18] + x_S[19] = 0
 c_SOC_dynamic[20]: - 0.2325 x_chg[20] + 0.2688172043010753 x_dis[20]
   - x_S[19] + x_S[20] = 0
 c_SOC_dynamic[21]: - 0.2325 x_chg[21] + 0.2688172043010753 x_dis[21]
   - x_S[20] + x_S[21] = 0
 c_SOC_dynamic[22]: - 0.2325 x_chg[22] + 0.2688172043010753 x_dis[22]
   - x_S[21] + x_S[22] = 0
 c_SOC_dynamic[23]: - 0.2325 x_chg[23] + 0.2688172043010753 x_dis[23]
   - x_S[22] + x_S[23] = 0
 c_SOC_dynamic[24]: - 0.2325 x_chg[24] + 0.2688172043010753 x_dis[24]
   - x_S[23] + x_S[24] = 0
 c_SOC_dynamic[25]: - 0.2325 x_chg[25] + 0.2688172043010753 x_dis[25]
   - x_S[24] + x_S[25] = 0
 c_SOC_dynamic[26]: - 0.2325 x_chg[26] + 0.2688172043010753 x_dis[26]
   - x_S[25] + x_S[26] = 0
 c_SOC_dynamic[27]: - 0.2325 x_chg[27] + 0.2688172043010753 x_dis[27]
   - x_S[26] + x_S[27] = 0
 c_SOC_dynamic[28]: - 0.2325 x_chg[28] + 0.2688172043010753 x_dis[28]
   - x_S[27] + x_S[28] = 0
 c_SOC_dynamic[29]: - 0.2325 x_chg[29] + 0.2688172043010753 x_dis[29]
   - x_S[28] + x_S[29] = 0
 c_SOC_dynamic[30]: - 0.2325 x_chg[30] + 0.2688172043010753 x_dis[30]
   - x_S[29] + x_S[30] = 0
 c_SOC_dynamic[31]: - 0.2325 x_chg[31] + 0.2688172043010753 x_dis[31]
   - x_S[30] + x_S[31] = 0
 c_SOC_dynamic[32]: - 0.2325 x_chg[32] + 0.2688172043010753 x_dis[32]
   - x_S[31] + x_S[32] = 0
 c_SOC_dynamic[33]: - 0.2325 x_chg[33] + 0.2688172043010753 x_dis[33]
   - x_S[32] + x_S[33] = 0
 c_SOC_dynamic[34]: - 0.2325 x_chg[34] + 0.2688172043010753 x_dis[34]
   - x_S[33] + x_S[34] = 0
 c_SOC_dynamic[35]: - 0.2325 x_chg[35] + 0.2688172043010753 x_dis[35]
   - x_S[34] + x_S[35] = 0
 c_SOC_dynamic[36]: - 0.2325 x_chg[36] + 0.2688172043010753 x_dis[36]
   - x_S[35] + x_S[36] = 0
 c_SOC_dynamic[37]: - 0.2325 x_chg[37] + 0.2688172043010753 x_dis[37]
   - x_S[36] + x_S[37] = 0
 c_SOC_dynamic[38]: - 0.2325 x_chg[38] + 0.2688172043010753 x_dis[38]
   - x_S[37] + x_S[38] = 0
 c_SOC_dynamic[39]: - 0.2325 x_chg[39] + 0.2688172043010753 x_dis[39]
   - x_S[38] + x_S[39] = 0
 c_SOC_dynamic[40]: - 0.2325 x_chg[40] + 0.2688172043010753 x_dis[40]
   - x_S[39] + x_S[40] = 0
 c_SOC_dynamic[41]: - 0.2325 x_chg[41] + 0.2688172043010753 x_dis[41]
   - x_S[40] + x_S[41] = 0
 c_SOC_dynamic[42]: - 0.2325 x_chg[42] + 0.2688172043010753 x_dis[42]
   - x_S[41] + x_S[42] = 0
 c_SOC_dynamic[43]: - 0.2325 x_chg[43] + 0.2688172043010753 x_dis[43]
   - x_S[42] + x_S[43] = 0
 c_SOC_dynamic[44]: - 0.2325 x_chg[44] + 0.2688172043010753 x_dis[44]
   - x_S[43] + x_S[44] = 0
 c_SOC_dynamic[45]: - 0.2325 x_chg[45] + 0.2688172043010753 x_dis[45]
   - x_S[44] + x_S[45] = 0
 c_SOC_dynamic[46]: - 0.2325 x_chg[46] + 0.2688172043010753 x_dis[46]
   - x_S[45] + x_S[46] = 0
 c_SOC_dynamic[47]: - 0.2325 x_chg[47] + 0.2688172043010753 x_dis[47]
   - x_S[46] + x_S[47] = 0
 c_SOC_dynamic[48]: - 0.2325 x_chg[48] + 0.2688172043010753 x_dis[48]
   - x_S[47] + x_S[48] = 0
 c_SOC_dynamic[49]: - 0.2325 x_chg[49] + 0.2688172043010753 x_dis[49]
   - x_S[48] + x_S[49] = 0
 c_SOC_dynamic[50]: - 0.2325 x_chg[50] + 0.2688172043010753 x_dis[50]
   - x_S[49] + x_S[50] = 0
 c_SOC_dynamic[51]: - 0.2325 x_chg[51] + 0.2688172043010753 x_dis[51]
   - x_S[50] + x_S[51] = 0
 c_SOC_dynamic[52]: - 0.2325 x_chg[52] + 0.2688172043010753 x_dis[52]
   - x_S[51] + x_S[52] = 0
 c_SOC_dynamic[53]: - 0.2325 x_chg[53] + 0.2688172043010753 x_dis[53]
   - x_S[52] + x_S[53] = 0
 c_SOC_dynamic[54]: - 0.2325 x_chg[54] + 0.2688172043010753 x_dis[54]
   - x_S[53] + x_S[54] = 0
 c_SOC_dynamic[55]: - 0.2325 x_chg[55] + 0.2688172043010753 x_dis[55]
   - x_S[54] + x_S[55] = 0
 c_SOC_dynamic[56]: - 0.2325 x_chg[56] + 0.2688172043010753 x_dis[56]
   - x_S[55] + x_S[56] = 0
 c_SOC_dynamic[57]: - 0.2325 x_chg[57] + 0.2688172043010753 x_dis[57]
   - x_S[56] + x_S[57] = 0
 c_SOC_dynamic[58]: - 0.2325 x_chg[58] + 0.2688172043010753 x_dis[58]
   - x_S[57] + x_S[58] = 0
 c_SOC_dynamic[59]: - 0.2325 x_chg[59] + 0.2688172043010753 x_dis[59]
   - x_S[58] + x_S[59] = 0
 c_SOC_dynamic[60]: - 0.2325 x_chg[60] + 0.2688172043010753 x_dis[60]
   - x_S[59] + x_S[60] = 0
 c_SOC_dynamic[61]: - 0.2325 x_chg[61] + 0.2688172043010753 x_dis[61]
   - x_S[60] + x_S[61] = 0
 c_SOC_dynamic[62]: - 0.2325 x_chg[62] + 0.2688172043010753 x_dis[62]
   - x_S[61] + x_S[62] = 0
 c_SOC_dynamic[63]: - 0.2325 x_chg[63] + 0.2688172043010753 x_dis[63]
   - x_S[62] + x_S[63] = 0
 c_SOC_dynamic[64]: - 0.2325 x_chg[64] + 0.2688172043010753 x_dis[64]
   - x_S[63] + x_S[64] = 0
 c_SOC_dynamic[65]: - 0.2325 x_chg[65] + 0.2688172043010753 x_dis[65]
   - x_S[64] + x_S[65] = 0
 c_SOC_dynamic[66]: - 0.2325 x_chg[66] + 0.2688172043010753 x_dis[66]
   - x_S[65] + x_S[66] = 0
 c_SOC_dynamic[67]: - 0.2325 x_chg[67] + 0.2688172043010753 x_dis[67]
   - x_S[66] + x_S[67] = 0
 c_SOC_dynamic[68]: - 0.2325 x_chg[68] + 0.2688172043010753 x_dis[68]
   - x_S[67] + x_S[68] = 0
 c_SOC_dynamic[69]: - 0.2325 x_chg[69] + 0.2688172043010753 x_dis[69]
   - x_S[68] + x_S[69] = 0
 c_SOC_dynamic[70]: - 0.2325 x_chg[70] + 0.2688172043010753 x_dis[70]
   - x_S[69] + x_S[70] = 0
 c_SOC_dynamic[71]: - 0.2325 x_chg[71] + 0.2688172043010753 x_dis[71]
   - x_S[70] + x_S[71] = 0
 c_SOC_dynamic[72]: - 0.2325 x_chg[72] + 0.2688172043010753 x_dis[72]
   - x_S[71] + x_S[72] = 0
 c_SOC_dynamic[73]: - 0.2325 x_chg[73] + 0.2688172043010753 x_dis[73]
   - x_S[72] + x_S[73] = 0
 c_SOC_dynamic[74]: - 0.2325 x_chg[74] + 0.2688172043010753 x_dis[74]
   - x_S[73] + x_S[74] = 0
 c_SOC_dynamic[75]: - 0.2325 x_chg[75] + 0.2688172043010753 x_dis[75]
   - x_S[74] + x_S[75] = 0
 c_SOC_dynamic[76]: - 0.2325 x_chg[76] + 0.2688172043010753 x_dis[76]
   - x_S[75] + x_S[76] = 0
 c_SOC_dynamic[77]: - 0.2325 x_chg[77] + 0.2688172043010753 x_dis[77]
   - x_S[76] + x_S[77] = 0
 c_SOC_dynamic[78]: - 0.2325 x_chg[78] + 0.2688172043010753 x_dis[78]
   - x_S[77] + x_S[78] = 0
 c_SOC_dynamic[79]: - 0.2325 x_chg[79] + 0.2688172043010753 x_dis[79]
   - x_S[78] + x_S[79] = 0
 c_SOC_dynamic[80]: - 0.2325 x_chg[80] + 0.2688172043010753 x_dis[80]
   - x_S[79] + x_S[80] = 0
 c_SOC_dynamic[81]: - 0.2325 x_chg[81] + 0.2688172043010753 x_dis[81]
   - x_S[80] + x_S[81] = 0
 c_SOC_dynamic[82]: - 0.2325 x_chg[82] + 0.2688172043010753 x_dis[82]
   - x_S[81] + x_S[82] = 0
 c_SOC_dynamic[83]: - 0.2325 x_chg[83] + 0.2688172043010753 x_dis[83]
   - x_S[82] + x_S[83] = 0
 c_SOC_dynamic[84]: - 0.2325 x_chg[84] + 0.2688172043010753 x_dis[84]
   - x_S[83] + x_S[84] = 0
 c_SOC_dynamic[85]: - 0.2325 x_chg[85] + 0.2688172043010753 x_dis[85]
   - x_S[84] + x_S[85] = 0
 c_SOC_dynamic[86]: - 0.2325 x_chg[86] + 0.2688172043010753 x_dis[86]
   - x_S[85] + x_S[86] = 0
 c_SOC_dynamic[87]: - 0.2325 x_chg[87] + 0.2688172043010753 x_dis[87]
   - x_S[86] + x_S[87] = 0
 c_SOC_dynamic[88]: - 0.2325 x_chg[88] + 0.2688172043010753 x_dis[88]
   - x_S[87] + x_S[88] = 0
 c_SOC_dynamic[89]: - 0.2325 x_chg[89] + 0.2688172043010753 x_dis[89]
   - x_S[88] + x_S[89] = 0
 c_SOC_dynamic[90]: - 0.2325 x_chg[90] + 0.2688172043010753 x_dis[90]
   - x_S[89] + x_S[90] = 0
 c_SOC_dynamic[91]: - 0.2325 x_chg[91] + 0.2688172043010753 x_dis[91]
   - x_S[90] + x_S[91] = 0
 c_SOC_dynamic[92]: - 0.2325 x_chg[92] + 0.2688172043010753 x_dis[92]
   - x_S[91] + x_S[92] = 0
 c_SOC_dynamic[93]: - 0.2325 x_chg[93] + 0.2688172043010753 x_dis[93]
   - x_S[92] + x_S[93] = 0
 c_SOC_dynamic[94]: - 0.2325 x_chg[94] + 0.2688172043010753 x_dis[94]
   - x_S[93] + x_S[94] = 0
 c_SOC_dynamic[95]: - 0.2325 x_chg[95] + 0.2688172043010753 x_dis[95]
   - x_S[94] + x_S[95] = 0
 c_SOC_last: x_S[95] = 187.5
 c_power_balance[0]: x[0] - x_chg[0] + x_dis[0] + x_PV[0] - x_cut[0]
   - x_load[0] = 0
 c_power_balance[1]: x[1] - x_chg[1] + x_dis[1] + x_PV[1] - x_cut[1]
   - x_load[1] = 0
 c_power_balance[2]: x[2] - x_chg[2] + x_dis[2] + x_PV[2] - x_cut[2]
   - x_load[2] = 0
 c_power_balance[3]: x[3] - x_chg[3] + x_dis[3] + x_PV[3] - x_cut[3]
   - x_load[3] = 0
 c_power_balance[4]: x[4] - x_chg[4] + x_dis[4] + x_PV[4] - x_cut[4]
   - x_load[4] = 0
 c_power_balance[5]: x[5] - x_chg[5] + x_dis[5] + x_PV[5] - x_cut[5]
   - x_load[5] = 0
 c_power_balance[6]: x[6] - x_chg[6] + x_dis[6] + x_PV[6] - x_cut[6]
   - x_load[6] = 0
 c_power_balance[7]: x[7] - x_chg[7] + x_dis[7] + x_PV[7] - x_cut[7]
   - x_load[7] = 0
 c_power_balance[8]: x[8] - x_chg[8] + x_dis[8] + x_PV[8] - x_cut[8]
   - x_load[8] = 0
 c_power_balance[9]: x[9] - x_chg[9] + x_dis[9] + x_PV[9] - x_cut[9]
   - x_load[9] = 0
 c_power_balance[10]: x[10] - x_chg[10] + x_dis[10] + x_PV[10] - x_cut[10]
   - x_load[10] = 0
 c_power_balance[11]: x[11] - x_chg[11] + x_dis[11] + x_PV[11] - x_cut[11]
   - x_load[11] = 0
 c_power_balance[12]: x[12] - x_chg[12] + x_dis[12] + x_PV[12] - x_cut[12]
   - x_load[12] = 0
 c_power_balance[13]: x[13] - x_chg[13] + x_dis[13] + x_PV[13] - x_cut[13]
   - x_load[13] = 0
 c_power_balance[14]: x[14] - x_chg[14] + x_dis[14] + x_PV[14] - x_cut[14]
   - x_load[14] = 0
 c_power_balance[15]: x[15] - x_chg[15] + x_dis[15] + x_PV[15] - x_cut[15]
   - x_load[15] = 0
 c_power_balance[16]: x[16] - x_chg[16] + x_dis[16] + x_PV[16] - x_cut[16]
   - x_load[16] = 0
 c_power_balance[17]: x[17] - x_chg[17] + x_dis[17] + x_PV[17] - x_cut[17]
   - x_load[17] = 0
 c_power_balance[18]: x[18] - x_chg[18] + x_dis[18] + x_PV[18] - x_cut[18]
   - x_load[18] = 0
 c_power_balance[19]: x[19] - x_chg[19] + x_dis[19] + x_PV[19] - x_cut[19]
   - x_load[19] = 0
 c_power_balance[20]: x[20] - x_chg[20] + x_dis[20] + x_PV[20] - x_cut[20]
   - x_load[20] = 0
 c_power_balance[21]: x[21] - x_chg[21] + x_dis[21] + x_PV[21] - x_cut[21]
   - x_load[21] = 0
 c_power_balance[22]: x[22] - x_chg[22] + x_dis[22] + x_PV[22] - x_cut[22]
   - x_load[22] = 0
 c_power_balance[23]: x[23] - x_chg[23] + x_dis[23] + x_PV[23] - x_cut[23]
   - x_load[23] = 0
 c_power_balance[24]: x[24] - x_chg[24] + x_dis[24] + x_PV[24] - x_cut[24]
   - x_load[24] = 0
 c_power_balance[25]: x[25] - x_chg[25] + x_dis[25] + x_PV[25] - x_cut[25]
   - x_load[25] = 0
 c_power_balance[26]: x[26] - x_chg[26] + x_dis[26] + x_PV[26] - x_cut[26]
   - x_load[26] = 0
 c_power_balance[27]: x[27] - x_chg[27] + x_dis[27] + x_PV[27] - x_cut[27]
   - x_load[27] = 0
 c_power_balance[28]: x[28] - x_chg[28] + x_dis[28] + x_PV[28] - x_cut[28]
   - x_load[28] = 0
 c_power_balance[29]: x[29] - x_chg[29] + x_dis[29] + x_PV[29] - x_cut[29]
   - x_load[29] = 0
 c_power_balance[30]: x[30] - x_chg[30] + x_dis[30] + x_PV[30] - x_cut[30]
   - x_load[30] = 0
 c_power_balance[31]: x[31] - x_chg[31] + x_dis[31] + x_PV[31] - x_cut[31]
   - x_load[31] = 0
 c_power_balance[32]: x[32] - x_chg[32] + x_dis[32] + x_PV[32] - x_cut[32]
   - x_load[32] = 0
 c_power_balance[33]: x[33] - x_chg[33] + x_dis[33] + x_PV[33] - x_cut[33]
   - x_load[33] = 0
 c_power_balance[34]: x[34] - x_chg[34] + x_dis[34] + x_PV[34] - x_cut[34]
   - x_load[34] = 0
 c_power_balance[35]: x[35] - x_chg[35] + x_dis[35] + x_PV[35] - x_cut[35]
   - x_load[35] = 0
 c_power_balance[36]: x[36] - x_chg[36] + x_dis[36] + x_PV[36] - x_cut[36]
   - x_load[36] = 0
 c_power_balance[37]: x[37] - x_chg[37] + x_dis[37] + x_PV[37] - x_cut[37]
   - x_load[37] = 0
 c_power_balance[38]: x[38] - x_chg[38] + x_dis[38] + x_PV[38] - x_cut[38]
   - x_load[38] = 0
 c_power_balance[39]: x[39] - x_chg[39] + x_dis[39] + x_PV[39] - x_cut[39]
   - x_load[39] = 0
 c_power_balance[40]: x[40] - x_chg[40] + x_dis[40] + x_PV[40] - x_cut[40]
   - x_load[40] = 0
 c_power_balance[41]: x[41] - x_chg[41] + x_dis[41] + x_PV[41] - x_cut[41]
   - x_load[41] = 0
 c_power_balance[42]: x[42] - x_chg[42] + x_dis[42] + x_PV[42] - x_cut[42]
   - x_load[42] = 0
 c_power_balance[43]: x[43] - x_chg[43] + x_dis[43] + x_PV[43] - x_cut[43]
   - x_load[43] = 0
 c_power_balance[44]: x[44] - x_chg[44] + x_dis[44] + x_PV[44] - x_cut[44]
   - x_load[44] = 0
 c_power_balance[45]: x[45] - x_chg[45] + x_dis[45] + x_PV[45] - x_cut[45]
   - x_load[45] = 0
 c_power_balance[46]: x[46] - x_chg[46] + x_dis[46] + x_PV[46] - x_cut[46]
   - x_load[46] = 0
 c_power_balance[47]: x[47] - x_chg[47] + x_dis[47] + x_PV[47] - x_cut[47]
   - x_load[47] = 0
 c_power_balance[48]: x[48] - x_chg[48] + x_dis[48] + x_PV[48] - x_cut[48]
   - x_load[48] = 0
 c_power_balance[49]: x[49] - x_chg[49] + x_dis[49] + x_PV[49] - x_cut[49]
   - x_load[49] = 0
 c_power_balance[50]: x[50] - x_chg[50] + x_dis[50] + x_PV[50] - x_cut[50]
   - x_load[50] = 0
 c_power_balance[51]: x[51] - x_chg[51] + x_dis[51] + x_PV[51] - x_cut[51]
   - x_load[51] = 0
 c_power_balance[52]: x[52] - x_chg[52] + x_dis[52] + x_PV[52] - x_cut[52]
   - x_load[52] = 0
 c_power_balance[53]: x[53] - x_chg[53] + x_dis[53] + x_PV[53] - x_cut[53]
   - x_load[53] = 0
 c_power_balance[54]: x[54] - x_chg[54] + x_dis[54] + x_PV[54] - x_cut[54]
   - x_load[54] = 0
 c_power_balance[55]: x[55] - x_chg[55] + x_dis[55] + x_PV[55] - x_cut[55]
   - x_load[55] = 0
 c_power_balance[56]: x[56] - x_chg[56] + x_dis[56] + x_PV[56] - x_cut[56]
   - x_load[56] = 0
 c_power_balance[57]: x[57] - x_chg[57] + x_dis[57] + x_PV[57] - x_cut[57]
   - x_load[57] = 0
 c_power_balance[58]: x[58] - x_chg[58] + x_dis[58] + x_PV[58] - x_cut[58]
   - x_load[58] = 0
 c_power_balance[59]: x[59] - x_chg[59] + x_dis[59] + x_PV[59] - x_cut[59]
   - x_load[59] = 0
 c_power_balance[60]: x[60] - x_chg[60] + x_dis[60] + x_PV[60] - x_cut[60]
   - x_load[60] = 0
 c_power_balance[61]: x[61] - x_chg[61] + x_dis[61] + x_PV[61] - x_cut[61]
   - x_load[61] = 0
 c_power_balance[62]: x[62] - x_chg[62] + x_dis[62] + x_PV[62] - x_cut[62]
   - x_load[62] = 0
 c_power_balance[63]: x[63] - x_chg[63] + x_dis[63] + x_PV[63] - x_cut[63]
   - x_load[63] = 0
 c_power_balance[64]: x[64] - x_chg[64] + x_dis[64] + x_PV[64] - x_cut[64]
   - x_load[64] = 0
 c_power_balance[65]: x[65] - x_chg[65] + x_dis[65] + x_PV[65] - x_cut[65]
   - x_load[65] = 0
 c_power_balance[66]: x[66] - x_chg[66] + x_dis[66] + x_PV[66] - x_cut[66]
   - x_load[66] = 0
 c_power_balance[67]: x[67] - x_chg[67] + x_dis[67] + x_PV[67] - x_cut[67]
   - x_load[67] = 0
 c_power_balance[68]: x[68] - x_chg[68] + x_dis[68] + x_PV[68] - x_cut[68]
   - x_load[68] = 0
 c_power_balance[69]: x[69] - x_chg[69] + x_dis[69] + x_PV[69] - x_cut[69]
   - x_load[69] = 0
 c_power_balance[70]: x[70] - x_chg[70] + x_dis[70] + x_PV[70] - x_cut[70]
   - x_load[70] = 0
 c_power_balance[71]: x[71] - x_chg[71] + x_dis[71] + x_PV[71] - x_cut[71]
   - x_load[71] = 0
 c_power_balance[72]: x[72] - x_chg[72] + x_dis[72] + x_PV[72] - x_cut[72]
   - x_load[72] = 0
 c_power_balance[73]: x[73] - x_chg[73] + x_dis[73] + x_PV[73] - x_cut[73]
   - x_load[73] = 0
 c_power_balance[74]: x[74] - x_chg[74] + x_dis[74] + x_PV[74] - x_cut[74]
   - x_load[74] = 0
 c_power_balance[75]: x[75] - x_chg[75] + x_dis[75] + x_PV[75] - x_cut[75]
   - x_load[75] = 0
 c_power_balance[76]: x[76] - x_chg[76] + x_dis[76] + x_PV[76] - x_cut[76]
   - x_load[76] = 0
 c_power_balance[77]: x[77] - x_chg[77] + x_dis[77] + x_PV[77] - x_cut[77]
   - x_load[77] = 0
 c_power_balance[78]: x[78] - x_chg[78] + x_dis[78] + x_PV[78] - x_cut[78]
   - x_load[78] = 0
 c_power_balance[79]: x[79] - x_chg[79] + x_dis[79] + x_PV[79] - x_cut[79]
   - x_load[79] = 0
 c_power_balance[80]: x[80] - x_chg[80] + x_dis[80] + x_PV[80] - x_cut[80]
   - x_load[80] = 0
 c_power_balance[81]: x[81] - x_chg[81] + x_dis[81] + x_PV[81] - x_cut[81]
   - x_load[81] = 0
 c_power_balance[82]: x[82] - x_chg[82] + x_dis[82] + x_PV[82] - x_cut[82]
   - x_load[82] = 0
 c_power_balance[83]: x[83] - x_chg[83] + x_dis[83] + x_PV[83] - x_cut[83]
   - x_load[83] = 0
 c_power_balance[84]: x[84] - x_chg[84] + x_dis[84] + x_PV[84] - x_cut[84]
   - x_load[84] = 0
 c_power_balance[85]: x[85] - x_chg[85] + x_dis[85] + x_PV[85] - x_cut[85]
   - x_load[85] = 0
 c_power_balance[86]: x[86] - x_chg[86] + x_dis[86] + x_PV[86] - x_cut[86]
   - x_load[86] = 0
 c_power_balance[87]: x[87] - x_chg[87] + x_dis[87] + x_PV[87] - x_cut[87]
   - x_load[87] = 0
 c_power_balance[88]: x[88] - x_chg[88] + x_dis[88] + x_PV[88] - x_cut[88]
   - x_load[88] = 0
 c_power_balance[89]: x[89] - x_chg[89] + x_dis[89] + x_PV[89] - x_cut[89]
   - x_load[89] = 0
 c_power_balance[90]: x[90] - x_chg[90] + x_dis[90] + x_PV[90] - x_cut[90]
   - x_load[90] = 0
 c_power_balance[91]: x[91] - x_chg[91] + x_dis[91] + x_PV[91] - x_cut[91]
   - x_load[91] = 0
 c_power_balance[92]: x[92] - x_chg[92] + x_dis[92] + x_PV[92] - x_cut[92]
   - x_load[92] = 0
 c_power_balance[93]: x[93] - x_chg[93] + x_dis[93] + x_PV[93] - x_cut[93]
   - x_load[93] = 0
 c_power_balance[94]: x[94] - x_chg[94] + x_dis[94] + x_PV[94] - x_cut[94]
   - x_load[94] = 0
 c_power_balance[95]: x[95] - x_chg[95] + x_dis[95] + x_PV[95] - x_cut[95]
   - x_load[95] = 0
 c_PV_output[0]: x_PV[0] = 0
 c_PV_output[1]: x_PV[1] = 0
 c_PV_output[2]: x_PV[2] = 0
 c_PV_output[3]: x_PV[3] = 0
 c_PV_output[4]: x_PV[4] = 0
 c_PV_output[5]: x_PV[5] = 0
 c_PV_output[6]: x_PV[6] = 0
 c_PV_output[7]: x_PV[7] = 0
 c_PV_output[8]: x_PV[8] = 0
 c_PV_output[9]: x_PV[9] = 0
 c_PV_output[10]: x_PV[10] = 0
 c_PV_output[11]: x_PV[11] = 0
 c_PV_output[12]: x_PV[12] = 0
 c_PV_output[13]: x_PV[13] = 0
 c_PV_output[14]: x_PV[14] = 0
 c_PV_output[15]: x_PV[15] = 0
 c_PV_output[16]: x_PV[16] = 0
 c_PV_output[17]: x_PV[17] = 0
 c_PV_output[18]: x_PV[18] = 0
 c_PV_output[19]: x_PV[19] = 0
 c_PV_output[20]: x_PV[20] = 0
 c_PV_output[21]: x_PV[21] = 0
 c_PV_output[22]: x_PV[22] = 0
 c_PV_output[23]: x_PV[23] = 0
 c_PV_output[24]: x_PV[24] = 0
 c_PV_output[25]: x_PV[25] = 0
 c_PV_output[26]: x_PV[26] = 0
 c_PV_output[27]: x_PV[27] = 0
 c_PV_output[28]: x_PV[28] = 0
 c_PV_output[29]: x_PV[29] = 0.6817549108937482
 c_PV_output[30]: x_PV[30] = 3.927929373725216
 c_PV_output[31]: x_PV[31] = 12.01863468178648
 c_PV_output[32]: x_PV[32] = 30.51313193721121
 c_PV_output[33]: x_PV[33] = 35.08499965125498
 c_PV_output[34]: x_PV[34] = 55.17332513737617
 c_PV_output[35]: x_PV[35] = 140.1686278350507
 c_PV_output[36]: x_PV[36] = 178.2836440688019
 c_PV_output[37]: x_PV[37] = 207.9424740079004
 c_PV_output[38]: x_PV[38] = 232.8479934620799
 c_PV_output[39]: x_PV[39] = 263.3768905490792
 c_PV_output[40]: x_PV[40] = 287.0985454425697
 c_PV_output[41]: x_PV[41] = 309.8822349462725
 c_PV_output[42]: x_PV[42] = 333.2777732532034
 c_PV_output[43]: x_PV[43] = 352.389781719958
 c_PV_output[44]: x_PV[44] = 375.7159161102198
 c_PV_output[45]: x_PV[45] = 394.9875487973123
 c_PV_output[46]: x_PV[46] = 409.7917531813302
 c_PV_output[47]: x_PV[47] = 428.101815569561
 c_PV_output[48]: x_PV[48] = 440.6464964382849
 c_PV_output[49]: x_PV[49] = 445.3786620241449
 c_PV_output[50]: x_PV[50] = 455.556472272189
 c_PV_output[51]: x_PV[51] = 460.5117236642729
 c_PV_output[52]: x_PV[52] = 464.00084206478
 c_PV_output[53]: x_PV[53] = 469.2390762828775
 c_PV_output[54]: x_PV[54] = 476.6880525993341
 c_PV_output[55]: x_PV[55] = 473.7111903963382
 c_PV_output[56]: x_PV[56] = 455.8462723276253
 c_PV_output[57]: x_PV[57] = 436.6431681623882
 c_PV_output[58]: x_PV[58] = 414.4684218590571
 c_PV_output[59]: x_PV[59] = 393.7351348159873
 c_PV_output[60]: x_PV[60] = 290.1728620584943
 c_PV_output[61]: x_PV[61] = 350.6164410277992
 c_PV_output[62]: x_PV[62] = 322.9022729394957
 c_PV_output[63]: x_PV[63] = 293.0620625079054
 c_PV_output[64]: x_PV[64] = 262.7833201991659
 c_PV_output[65]: x_PV[65] = 231.385172051662
 c_PV_output[66]: x_PV[66] = 199.9895743469131
 c_PV_output[67]: x_PV[67] = 165.7915647075124
 c_PV_output[68]: x_PV[68] = 132.5896733187977
 c_PV_output[69]: x_PV[69] = 98.15859347428747
 c_PV_output[70]: x_PV[70] = 40.61665929262848
 c_PV_output[71]: x_PV[71] = 30.04096046923865
 c_PV_output[72]: x_PV[72] = 16.20481073549318
 c_PV_output[73]: x_PV[73] = 2.626139118086846
 c_PV_output[74]: x_PV[74] = 0
 c_PV_output[75]: x_PV[75] = 0
 c_PV_output[76]: x_PV[76] = 0
 c_PV_output[77]: x_PV[77] = 0
 c_PV_output[78]: x_PV[78] = 0
 c_PV_output[79]: x_PV[79] = 0
 c_PV_output[80]: x_PV[80] = 0
 c_PV_output[81]: x_PV[81] = 0
 c_PV_output[82]: x_PV[82] = 0
 c_PV_output[83]: x_PV[83] = 0
 c_PV_output[84]: x_PV[84] = 0
 c_PV_output[85]: x_PV[85] = 0
 c_PV_output[86]: x_PV[86] = 0
 c_PV_output[87]: x_PV[87] = 0
 c_PV_output[88]: x_PV[88] = 0
 c_PV_output[89]: x_PV[89] = 0
 c_PV_output[90]: x_PV[90] = 0
 c_PV_output[91]: x_PV[91] = 0
 c_PV_output[92]: x_PV[92] = 0
 c_PV_output[93]: x_PV[93] = 0
 c_PV_output[94]: x_PV[94] = 0
 c_PV_output[95]: x_PV[95] = 0
 c_x_cut[0]: x_cut[0] <= 1e-05
 c_x_cut[1]: x_cut[1] <= 1e-05
 c_x_cut[2]: x_cut[2] <= 1e-05
 c_x_cut[3]: x_cut[3] <= 1e-05
 c_x_cut[4]: x_cut[4] <= 1e-05
 c_x_cut[5]: x_cut[5] <= 1e-05
 c_x_cut[6]: x_cut[6] <= 1e-05
 c_x_cut[7]: x_cut[7] <= 1e-05
 c_x_cut[8]: x_cut[8] <= 1e-05
 c_x_cut[9]: x_cut[9] <= 1e-05
 c_x_cut[10]: x_cut[10] <= 1e-05
 c_x_cut[11]: x_cut[11] <= 1e-05
 c_x_cut[12]: x_cut[12] <= 1e-05
 c_x_cut[13]: x_cut[13] <= 1e-05
 c_x_cut[14]: x_cut[14] <= 1e-05
 c_x_cut[15]: x_cut[15] <= 1e-05
 c_x_cut[16]: x_cut[16] <= 1e-05
 c_x_cut[17]: x_cut[17] <= 1e-05
 c_x_cut[18]: x_cut[18] <= 1e-05
 c_x_cut[19]: x_cut[19] <= 1e-05
 c_x_cut[20]: x_cut[20] <= 1e-05
 c_x_cut[21]: x_cut[21] <= 1e-05
 c_x_cut[22]: x_cut[22] <= 1e-05
 c_x_cut[23]: x_cut[23] <= 1e-05
 c_x_cut[24]: x_cut[24] <= 1e-05
 c_x_cut[25]: x_cut[25] <= 1e-05
 c_x_cut[26]: x_cut[26] <= 1e-05
 c_x_cut[27]: x_cut[27] <= 1e-05
 c_x_cut[28]: x_cut[28] <= 1e-05
 c_x_cut[29]: x_cut[29] <= 1e-05
 c_x_cut[30]: x_cut[30] <= 1e-05
 c_x_cut[31]: x_cut[31] <= 0.0253118027355328
 c_x_cut[32]: x_cut[32] <= 3.56637887912738
 c_x_cut[33]: x_cut[33] <= 6.066765780669403
 c_x_cut[34]: x_cut[34] <= 11.02108279071951
 c_x_cut[35]: x_cut[35] <= 40.14181803835542
 c_x_cut[36]: x_cut[36] <= 58.64215775270846
 c_x_cut[37]: x_cut[37] <= 71.18195246506995
 c_x_cut[38]: x_cut[38] <= 78.63593092742535
 c_x_cut[39]: x_cut[39] <= 103.9598003496597
 c_x_cut[40]: x_cut[40] <= 120.7964966037431
 c_x_cut[41]: x_cut[41] <= 130.8044015164348
 c_x_cut[42]: x_cut[42] <= 135.2361679921553
 c_x_cut[43]: x_cut[43] <= 157.5312789367545
 c_x_cut[44]: x_cut[44] <= 173.7538437201719
 c_x_cut[45]: x_cut[45] <= 179.6553866724427
 c_x_cut[46]: x_cut[46] <= 175.7944314299832
 c_x_cut[47]: x_cut[47] <= 200.7069986029999
 c_x_cut[48]: x_cut[48] <= 214.3541259015806
 c_x_cut[49]: x_cut[49] <= 214.3913137253857
 c_x_cut[50]: x_cut[50] <= 207.6329982380296
 c_x_cut[51]: x_cut[51] <= 222.9774754564774
 c_x_cut[52]: x_cut[52] <= 229.0993423760023
 c_x_cut[53]: x_cut[53] <= 226.4956731384092
 c_x_cut[54]: x_cut[54] <= 215.2602660261786
 c_x_cut[55]: x_cut[55] <= 222.2076689793372
 c_x_cut[56]: x_cut[56] <= 214.1577624899606
 c_x_cut[57]: x_cut[57] <= 196.5971352475246
 c_x_cut[58]: x_cut[58] <= 169.3389034902879
 c_x_cut[59]: x_cut[59] <= 164.9690439176622
 c_x_cut[60]: x_cut[60] <= 120.089511355573
 c_x_cut[61]: x_cut[61] <= 135.8592201117795
 c_x_cut[62]: x_cut[62] <= 107.9922319062659
 c_x_cut[63]: x_cut[63] <= 97.49370147574699
 c_x_cut[64]: x_cut[64] <= 82.66484505094192
 c_x_cut[65]: x_cut[65] <= 62.58093412347177
 c_x_cut[66]: x_cut[66] <= 37.35567075234511
 c_x_cut[67]: x_cut[67] <= 25.77433562208772
 c_x_cut[68]: x_cut[68] <= 12.87967427203721
 c_x_cut[69]: x_cut[69] <= 1e-05
 c_x_cut[70]: x_cut[70] <= 1e-05
 c_x_cut[71]: x_cut[71] <= 1e-05
 c_x_cut[72]: x_cut[72] <= 1e-05
 c_x_cut[73]: x_cut[73] <= 1e-05
 c_x_cut[74]: x_cut[74] <= 1e-05
 c_x_cut[75]: x_cut[75] <= 1e-05
 c_x_cut[76]: x_cut[76] <= 1e-05
 c_x_cut[77]: x_cut[77] <= 1e-05
 c_x_cut[78]: x_cut[78] <= 1e-05
 c_x_cut[79]: x_cut[79] <= 1e-05
 c_x_cut[80]: x_cut[80] <= 1e-05
 c_x_cut[81]: x_cut[81] <= 1e-05
 c_x_cut[82]: x_cut[82] <= 1e-05
 c_x_cut[83]: x_cut[83] <= 1e-05
 c_x_cut[84]: x_cut[84] <= 1e-05
 c_x_cut[85]: x_cut[85] <= 1e-05
 c_x_cut[86]: x_cut[86] <= 1e-05
 c_x_cut[87]: x_cut[87] <= 1e-05
 c_x_cut[88]: x_cut[88] <= 1e-05
 c_x_cut[89]: x_cut[89] <= 1e-05
 c_x_cut[90]: x_cut[90] <= 1e-05
 c_x_cut[91]: x_cut[91] <= 1e-05
 c_x_cut[92]: x_cut[92] <= 1e-05
 c_x_cut[93]: x_cut[93] <= 1e-05
 c_x_cut[94]: x_cut[94] <= 1e-05
 c_x_cut[95]: x_cut[95] <= 1e-05
 c_x_load_demand[0]: x_load[0] = 308.1600036621094
 c_x_load_demand[1]: x_load[1] = 306.4299926757813
 c_x_load_demand[2]: x_load[2] = 309.0700073242188
 c_x_load_demand[3]: x_load[3] = 307.8200073242188
 c_x_load_demand[4]: x_load[4] = 308.1600036621094
 c_x_load_demand[5]: x_load[5] = 310.4200134277344
 c_x_load_demand[6]: x_load[6] = 306.3800048828125
 c_x_load_demand[7]: x_load[7] = 309.9400024414063
 c_x_load_demand[8]: x_load[8] = 308.3999938964844
 c_x_load_demand[9]: x_load[9] = 309.5
 c_x_load_demand[10]: x_load[10] = 310.6600036621094
 c_x_load_demand[11]: x_load[11] = 309.8399963378906
 c_x_load_demand[12]: x_load[12] = 309.0199890136719
 c_x_load_demand[13]: x_load[13] = 311.6600036621094
 c_x_load_demand[14]: x_load[14] = 308.1099853515625
 c_x_load_demand[15]: x_load[15] = 311.9500122070313
 c_x_load_demand[16]: x_load[16] = 307.010009765625
 c_x_load_demand[17]: x_load[17] = 308.9800109863281
 c_x_load_demand[18]: x_load[18] = 308.1600036621094
 c_x_load_demand[19]: x_load[19] = 307.6300048828125
 c_x_load_demand[20]: x_load[20] = 308.7799987792969
 c_x_load_demand[21]: x_load[21] = 307.4400024414063
 c_x_load_demand[22]: x_load[22] = 308.0199890136719
 c_x_load_demand[23]: x_load[23] = 309.0700073242188
 c_x_load_demand[24]: x_load[24] = 306.5299987792969
 c_x_load_demand[25]: x_load[25] = 308.9299926757813
 c_x_load_demand[26]: x_load[26] = 305.5700073242188
 c_x_load_demand[27]: x_load[27] = 305.7099914550781
 c_x_load_demand[28]: x_load[28] = 308.3999938964844
 c_x_load_demand[29]: x_load[29] = 304.9400024414063
 c_x_load_demand[30]: x_load[30] = 306.3399963378906
 c_x_load_demand[31]: x_load[31] = 303.8399963378906
 c_x_load_demand[32]: x_load[32] = 302.3999938964844
 c_x_load_demand[33]: x_load[33] = 305.6600036621094
 c_x_load_demand[34]: x_load[34] = 303.8900146484375
 c_x_load_demand[35]: x_load[35] = 305.1799926757813
 c_x_load_demand[36]: x_load[36] = 306.5299987792969
 c_x_load_demand[37]: x_load[37] = 308.0599975585938
 c_x_load_demand[38]: x_load[38] = 307.3900146484375
 c_x_load_demand[39]: x_load[39] = 306.4800109863281
 c_x_load_demand[40]: x_load[40] = 307.25
 c_x_load_demand[41]: x_load[41] = 307.9200134277344
 c_x_load_demand[42]: x_load[42] = 305.1799926757813
 c_x_load_demand[43]: x_load[43] = 308.0599975585938
 c_x_load_demand[44]: x_load[44] = 307.3900146484375
 c_x_load_demand[45]: x_load[45] = 306.1400146484375
 c_x_load_demand[46]: x_load[46] = 307.7300109863281
 c_x_load_demand[47]: x_load[47] = 305.7099914550781
 c_x_load_demand[48]: x_load[48] = 306.5299987792969
 c_x_load_demand[49]: x_load[49] = 306.9599914550781
 c_x_load_demand[50]: x_load[50] = 305.3800048828125
 c_x_load_demand[51]: x_load[51] = 307.7799987792969
 c_x_load_demand[52]: x_load[52] = 306.1900024414063
 c_x_load_demand[53]: x_load[53] = 306.8200073242188
 c_x_load_demand[54]: x_load[54] = 307.010009765625
 c_x_load_demand[55]: x_load[55] = 305.9500122070313
 c_x_load_demand[56]: x_load[56] = 306.4299926757813
 c_x_load_demand[57]: x_load[57] = 308.739990234375
 c_x_load_demand[58]: x_load[58] = 306.7699890136719
 c_x_load_demand[59]: x_load[59] = 309.3099975585938
 c_x_load_demand[60]: x_load[60] = 305.4700012207031
 c_x_load_demand[61]: x_load[61] = 307.6300048828125
 c_x_load_demand[62]: x_load[62] = 305.5700073242188
 c_x_load_demand[63]: x_load[63] = 306.5299987792969
 c_x_load_demand[64]: x_load[64] = 306.2900085449219
 c_x_load_demand[65]: x_load[65] = 304.2200012207031
 c_x_load_demand[66]: x_load[66] = 303.0199890136719
 c_x_load_demand[67]: x_load[67] = 304.3699951171875
 c_x_load_demand[68]: x_load[68] = 303.3099975585938
 c_x_load_demand[69]: x_load[69] = 305.4700012207031
 c_x_load_demand[70]: x_load[70] = 304.989990234375
 c_x_load_demand[71]: x_load[71] = 304.7999877929688
 c_x_load_demand[72]: x_load[72] = 306.1400146484375
 c_x_load_demand[73]: x_load[73] = 307.7300109863281
 c_x_load_demand[74]: x_load[74] = 304.8999938964844
 c_x_load_demand[75]: x_load[75] = 304.7999877929688
 c_x_load_demand[76]: x_load[76] = 303.7000122070313
 c_x_load_demand[77]: x_load[77] = 304.0299987792969
 c_x_load_demand[78]: x_load[78] = 303.2200012207031
 c_x_load_demand[79]: x_load[79] = 303.6000061035156
 c_x_load_demand[80]: x_load[80] = 302.9800109863281
 c_x_load_demand[81]: x_load[81] = 302.5400085449219
 c_x_load_demand[82]: x_load[82] = 303.6000061035156
 c_x_load_demand[83]: x_load[83] = 302.260009765625
 c_x_load_demand[84]: x_load[84] = 303.2200012207031
 c_x_load_demand[85]: x_load[85] = 302.7799987792969
 c_x_load_demand[86]: x_load[86] = 302.6400146484375
 c_x_load_demand[87]: x_load[87] = 303.5499877929688
 c_x_load_demand[88]: x_load[88] = 302.8299865722656
 c_x_load_demand[89]: x_load[89] = 303.7000122070313
 c_x_load_demand[90]: x_load[90] = 304.0299987792969
 c_x_load_demand[91]: x_load[91] = 303.5
 c_x_load_demand[92]: x_load[92] = 305.4700012207031
 c_x_load_demand[93]: x_load[93] = 304.6600036621094
 c_x_load_demand[94]: x_load[94] = 305.8999938964844
 c_x_load_demand[95]: x_load[95] = 306
 c_cost_fuel_res[0]: - 0.2 x_pos[0] - 0.15 x_neg[0] + x_cost_res[0] = 0
 c_cost_fuel_res[1]: - 0.2 x_pos[1] - 0.15 x_neg[1] + x_cost_res[1] = 0
 c_cost_fuel_res[2]: - 0.2 x_pos[2] - 0.15 x_neg[2] + x_cost_res[2] = 0
 c_cost_fuel_res[3]: - 0.2 x_pos[3] - 0.15 x_neg[3] + x_cost_res[3] = 0
 c_cost_fuel_res[4]: - 0.2 x_pos[4] - 0.15 x_neg[4] + x_cost_res[4] = 0
 c_cost_fuel_res[5]: - 0.2 x_pos[5] - 0.15 x_neg[5] + x_cost_res[5] = 0
 c_cost_fuel_res[6]: - 0.2 x_pos[6] - 0.15 x_neg[6] + x_cost_res[6] = 0
 c_cost_fuel_res[7]: - 0.2 x_pos[7] - 0.15 x_neg[7] + x_cost_res[7] = 0
 c_cost_fuel_res[8]: - 0.2 x_pos[8] - 0.15 x_neg[8] + x_cost_res[8] = 0
 c_cost_fuel_res[9]: - 0.2 x_pos[9] - 0.15 x_neg[9] + x_cost_res[9] = 0
 c_cost_fuel_res[10]: - 0.2 x_pos[10] - 0.15 x_neg[10] + x_cost_res[10]
   = 0
 c_cost_fuel_res[11]: - 0.2 x_pos[11] - 0.15 x_neg[11] + x_cost_res[11]
   = 0
 c_cost_fuel_res[12]: - 0.2 x_pos[12] - 0.15 x_neg[12] + x_cost_res[12]
   = 0
 c_cost_fuel_res[13]: - 0.2 x_pos[13] - 0.15 x_neg[13] + x_cost_res[13]
   = 0
 c_cost_fuel_res[14]: - 0.2 x_pos[14] - 0.15 x_neg[14] + x_cost_res[14]
   = 0
 c_cost_fuel_res[15]: - 0.2 x_pos[15] - 0.15 x_neg[15] + x_cost_res[15]
   = 0
 c_cost_fuel_res[16]: - 0.2 x_pos[16] - 0.15 x_neg[16] + x_cost_res[16]
   = 0
 c_cost_fuel_res[17]: - 0.2 x_pos[17] - 0.15 x_neg[17] + x_cost_res[17]
   = 0
 c_cost_fuel_res[18]: - 0.2 x_pos[18] - 0.15 x_neg[18] + x_cost_res[18]
   = 0
 c_cost_fuel_res[19]: - 0.2 x_pos[19] - 0.15 x_neg[19] + x_cost_res[19]
   = 0
 c_cost_fuel_res[20]: - 0.2 x_pos[20] - 0.15 x_neg[20] + x_cost_res[20]
   = 0
 c_cost_fuel_res[21]: - 0.2 x_pos[21] - 0.15 x_neg[21] + x_cost_res[21]
   = 0
 c_cost_fuel_res[22]: - 0.2 x_pos[22] - 0.15 x_neg[22] + x_cost_res[22]
   = 0
 c_cost_fuel_res[23]: - 0.2 x_pos[23] - 0.15 x_neg[23] + x_cost_res[23]
   = 0
 c_cost_fuel_res[24]: - 0.2 x_pos[24] - 0.15 x_neg[24] + x_cost_res[24]
   = 0
 c_cost_fuel_res[25]: - 0.2 x_pos[25] - 0.15 x_neg[25] + x_cost_res[25]
   = 0
 c_cost_fuel_res[26]: - 0.2 x_pos[26] - 0.15 x_neg[26] + x_cost_res[26]
   = 0
 c_cost_fuel_res[27]: - 0.2 x_pos[27] - 0.15 x_neg[27] + x_cost_res[27]
   = 0
 c_cost_fuel_res[28]: - 0.2 x_pos[28] - 0.15 x_neg[28] + x_cost_res[28]
   = 0
 c_cost_fuel_res[29]: - 0.2 x_pos[29] - 0.15 x_neg[29] + x_cost_res[29]
   = 0
 c_cost_fuel_res[30]: - 0.2 x_pos[30] - 0.15 x_neg[30] + x_cost_res[30]
   = 0
 c_cost_fuel_res[31]: - 0.2 x_pos[31] - 0.15 x_neg[31] + x_cost_res[31]
   = 0
 c_cost_fuel_res[32]: - 0.2 x_pos[32] - 0.15 x_neg[32] + x_cost_res[32]
   = 0
 c_cost_fuel_res[33]: - 0.2 x_pos[33] - 0.15 x_neg[33] + x_cost_res[33]
   = 0
 c_cost_fuel_res[34]: - 0.2 x_pos[34] - 0.15 x_neg[34] + x_cost_res[34]
   = 0
 c_cost_fuel_res[35]: - 0.2 x_pos[35] - 0.15 x_neg[35] + x_cost_res[35]
   = 0
 c_cost_fuel_res[36]: - 0.2 x_pos[36] - 0.15 x_neg[36] + x_cost_res[36]
   = 0
 c_cost_fuel_res[37]: - 0.2 x_pos[37] - 0.15 x_neg[37] + x_cost_res[37]
   = 0
 c_cost_fuel_res[38]: - 0.2 x_pos[38] - 0.15 x_neg[38] + x_cost_res[38]
   = 0
 c_cost_fuel_res[39]: - 0.2 x_pos[39] - 0.15 x_neg[39] + x_cost_res[39]
   = 0
 c_cost_fuel_res[40]: - 0.2 x_pos[40] - 0.15 x_neg[40] + x_cost_res[40]
   = 0
 c_cost_fuel_res[41]: - 0.2 x_pos[41] - 0.15 x_neg[41] + x_cost_res[41]
   = 0
 c_cost_fuel_res[42]: - 0.2 x_pos[42] - 0.15 x_neg[42] + x_cost_res[42]
   = 0
 c_cost_fuel_res[43]: - 0.2 x_pos[43] - 0.15 x_neg[43] + x_cost_res[43]
   = 0
 c_cost_fuel_res[44]: - 0.2 x_pos[44] - 0.15 x_neg[44] + x_cost_res[44]
   = 0
 c_cost_fuel_res[45]: - 0.2 x_pos[45] - 0.15 x_neg[45] + x_cost_res[45]
   = 0
 c_cost_fuel_res[46]: - 0.2 x_pos[46] - 0.15 x_neg[46] + x_cost_res[46]
   = 0
 c_cost_fuel_res[47]: - 0.2 x_pos[47] - 0.15 x_neg[47] + x_cost_res[47]
   = 0
 c_cost_fuel_res[48]: - 0.2 x_pos[48] - 0.15 x_neg[48] + x_cost_res[48]
   = 0
 c_cost_fuel_res[49]: - 0.2 x_pos[49] - 0.15 x_neg[49] + x_cost_res[49]
   = 0
 c_cost_fuel_res[50]: - 0.2 x_pos[50] - 0.15 x_neg[50] + x_cost_res[50]
   = 0
 c_cost_fuel_res[51]: - 0.2 x_pos[51] - 0.15 x_neg[51] + x_cost_res[51]
   = 0
 c_cost_fuel_res[52]: - 0.2 x_pos[52] - 0.15 x_neg[52] + x_cost_res[52]
   = 0
 c_cost_fuel_res[53]: - 0.2 x_pos[53] - 0.15 x_neg[53] + x_cost_res[53]
   = 0
 c_cost_fuel_res[54]: - 0.2 x_pos[54] - 0.15 x_neg[54] + x_cost_res[54]
   = 0
 c_cost_fuel_res[55]: - 0.2 x_pos[55] - 0.15 x_neg[55] + x_cost_res[55]
   = 0
 c_cost_fuel_res[56]: - 0.2 x_pos[56] - 0.15 x_neg[56] + x_cost_res[56]
   = 0
 c_cost_fuel_res[57]: - 0.2 x_pos[57] - 0.15 x_neg[57] + x_cost_res[57]
   = 0
 c_cost_fuel_res[58]: - 0.2 x_pos[58] - 0.15 x_neg[58] + x_cost_res[58]
   = 0
 c_cost_fuel_res[59]: - 0.2 x_pos[59] - 0.15 x_neg[59] + x_cost_res[59]
   = 0
 c_cost_fuel_res[60]: - 0.2 x_pos[60] - 0.15 x_neg[60] + x_cost_res[60]
   = 0
 c_cost_fuel_res[61]: - 0.2 x_pos[61] - 0.15 x_neg[61] + x_cost_res[61]
   = 0
 c_cost_fuel_res[62]: - 0.2 x_pos[62] - 0.15 x_neg[62] + x_cost_res[62]
   = 0
 c_cost_fuel_res[63]: - 0.2 x_pos[63] - 0.15 x_neg[63] + x_cost_res[63]
   = 0
 c_cost_fuel_res[64]: - 0.2 x_pos[64] - 0.15 x_neg[64] + x_cost_res[64]
   = 0
 c_cost_fuel_res[65]: - 0.2 x_pos[65] - 0.15 x_neg[65] + x_cost_res[65]
   = 0
 c_cost_fuel_res[66]: - 0.2 x_pos[66] - 0.15 x_neg[66] + x_cost_res[66]
   = 0
 c_cost_fuel_res[67]: - 0.2 x_pos[67] - 0.15 x_neg[67] + x_cost_res[67]
   = 0
 c_cost_fuel_res[68]: - 0.2 x_pos[68] - 0.15 x_neg[68] + x_cost_res[68]
   = 0
 c_cost_fuel_res[69]: - 0.2 x_pos[69] - 0.15 x_neg[69] + x_cost_res[69]
   = 0
 c_cost_fuel_res[70]: - 0.2 x_pos[70] - 0.15 x_neg[70] + x_cost_res[70]
   = 0
 c_cost_fuel_res[71]: - 0.2 x_pos[71] - 0.15 x_neg[71] + x_cost_res[71]
   = 0
 c_cost_fuel_res[72]: - 0.2 x_pos[72] - 0.15 x_neg[72] + x_cost_res[72]
   = 0
 c_cost_fuel_res[73]: - 0.2 x_pos[73] - 0.15 x_neg[73] + x_cost_res[73]
   = 0
 c_cost_fuel_res[74]: - 0.2 x_pos[74] - 0.15 x_neg[74] + x_cost_res[74]
   = 0
 c_cost_fuel_res[75]: - 0.2 x_pos[75] - 0.15 x_neg[75] + x_cost_res[75]
   = 0
 c_cost_fuel_res[76]: - 0.2 x_pos[76] - 0.15 x_neg[76] + x_cost_res[76]
   = 0
 c_cost_fuel_res[77]: - 0.2 x_pos[77] - 0.15 x_neg[77] + x_cost_res[77]
   = 0
 c_cost_fuel_res[78]: - 0.2 x_pos[78] - 0.15 x_neg[78] + x_cost_res[78]
   = 0
 c_cost_fuel_res[79]: - 0.2 x_pos[79] - 0.15 x_neg[79] + x_cost_res[79]
   = 0
 c_cost_fuel_res[80]: - 0.2 x_pos[80] - 0.15 x_neg[80] + x_cost_res[80]
   = 0
 c_cost_fuel_res[81]: - 0.2 x_pos[81] - 0.15 x_neg[81] + x_cost_res[81]
   = 0
 c_cost_fuel_res[82]: - 0.2 x_pos[82] - 0.15 x_neg[82] + x_cost_res[82]
   = 0
 c_cost_fuel_res[83]: - 0.2 x_pos[83] - 0.15 x_neg[83] + x_cost_res[83]
   = 0
 c_cost_fuel_res[84]: - 0.2 x_pos[84] - 0.15 x_neg[84] + x_cost_res[84]
   = 0
 c_cost_fuel_res[85]: - 0.2 x_pos[85] - 0.15 x_neg[85] + x_cost_res[85]
   = 0
 c_cost_fuel_res[86]: - 0.2 x_pos[86] - 0.15 x_neg[86] + x_cost_res[86]
   = 0
 c_cost_fuel_res[87]: - 0.2 x_pos[87] - 0.15 x_neg[87] + x_cost_res[87]
   = 0
 c_cost_fuel_res[88]: - 0.2 x_pos[88] - 0.15 x_neg[88] + x_cost_res[88]
   = 0
 c_cost_fuel_res[89]: - 0.2 x_pos[89] - 0.15 x_neg[89] + x_cost_res[89]
   = 0
 c_cost_fuel_res[90]: - 0.2 x_pos[90] - 0.15 x_neg[90] + x_cost_res[90]
   = 0
 c_cost_fuel_res[91]: - 0.2 x_pos[91] - 0.15 x_neg[91] + x_cost_res[91]
   = 0
 c_cost_fuel_res[92]: - 0.2 x_pos[92] - 0.15 x_neg[92] + x_cost_res[92]
   = 0
 c_cost_fuel_res[93]: - 0.2 x_pos[93] - 0.15 x_neg[93] + x_cost_res[93]
   = 0
 c_cost_fuel_res[94]: - 0.2 x_pos[94] - 0.15 x_neg[94] + x_cost_res[94]
   = 0
 c_cost_fuel_res[95]: - 0.2 x_pos[95] - 0.15 x_neg[95] + x_cost_res[95]
   = 0
 c_cost_OM_ESS[0]: - 0.3 x_chg[0] - 0.3 x_dis[0] + x_cost_ESS[0] = 0
 c_cost_OM_ESS[1]: - 0.3 x_chg[1] - 0.3 x_dis[1] + x_cost_ESS[1] = 0
 c_cost_OM_ESS[2]: - 0.3 x_chg[2] - 0.3 x_dis[2] + x_cost_ESS[2] = 0
 c_cost_OM_ESS[3]: - 0.3 x_chg[3] - 0.3 x_dis[3] + x_cost_ESS[3] = 0
 c_cost_OM_ESS[4]: - 0.3 x_chg[4] - 0.3 x_dis[4] + x_cost_ESS[4] = 0
 c_cost_OM_ESS[5]: - 0.3 x_chg[5] - 0.3 x_dis[5] + x_cost_ESS[5] = 0
 c_cost_OM_ESS[6]: - 0.3 x_chg[6] - 0.3 x_dis[6] + x_cost_ESS[6] = 0
 c_cost_OM_ESS[7]: - 0.3 x_chg[7] - 0.3 x_dis[7] + x_cost_ESS[7] = 0
 c_cost_OM_ESS[8]: - 0.3 x_chg[8] - 0.3 x_dis[8] + x_cost_ESS[8] = 0
 c_cost_OM_ESS[9]: - 0.3 x_chg[9] - 0.3 x_dis[9] + x_cost_ESS[9] = 0
 c_cost_OM_ESS[10]: - 0.3 x_chg[10] - 0.3 x_dis[10] + x_cost_ESS[10] = 0
 c_cost_OM_ESS[11]: - 0.3 x_chg[11] - 0.3 x_dis[11] + x_cost_ESS[11] = 0
 c_cost_OM_ESS[12]: - 0.3 x_chg[12] - 0.3 x_dis[12] + x_cost_ESS[12] = 0
 c_cost_OM_ESS[13]: - 0.3 x_chg[13] - 0.3 x_dis[13] + x_cost_ESS[13] = 0
 c_cost_OM_ESS[14]: - 0.3 x_chg[14] - 0.3 x_dis[14] + x_cost_ESS[14] = 0
 c_cost_OM_ESS[15]: - 0.3 x_chg[15] - 0.3 x_dis[15] + x_cost_ESS[15] = 0
 c_cost_OM_ESS[16]: - 0.3 x_chg[16] - 0.3 x_dis[16] + x_cost_ESS[16] = 0
 c_cost_OM_ESS[17]: - 0.3 x_chg[17] - 0.3 x_dis[17] + x_cost_ESS[17] = 0
 c_cost_OM_ESS[18]: - 0.3 x_chg[18] - 0.3 x_dis[18] + x_cost_ESS[18] = 0
 c_cost_OM_ESS[19]: - 0.3 x_chg[19] - 0.3 x_dis[19] + x_cost_ESS[19] = 0
 c_cost_OM_ESS[20]: - 0.3 x_chg[20] - 0.3 x_dis[20] + x_cost_ESS[20] = 0
 c_cost_OM_ESS[21]: - 0.3 x_chg[21] - 0.3 x_dis[21] + x_cost_ESS[21] = 0
 c_cost_OM_ESS[22]: - 0.3 x_chg[22] - 0.3 x_dis[22] + x_cost_ESS[22] = 0
 c_cost_OM_ESS[23]: - 0.3 x_chg[23] - 0.3 x_dis[23] + x_cost_ESS[23] = 0
 c_cost_OM_ESS[24]: - 0.3 x_chg[24] - 0.3 x_dis[24] + x_cost_ESS[24] = 0
 c_cost_OM_ESS[25]: - 0.3 x_chg[25] - 0.3 x_dis[25] + x_cost_ESS[25] = 0
 c_cost_OM_ESS[26]: - 0.3 x_chg[26] - 0.3 x_dis[26] + x_cost_ESS[26] = 0
 c_cost_OM_ESS[27]: - 0.3 x_chg[27] - 0.3 x_dis[27] + x_cost_ESS[27] = 0
 c_cost_OM_ESS[28]: - 0.3 x_chg[28] - 0.3 x_dis[28] + x_cost_ESS[28] = 0
 c_cost_OM_ESS[29]: - 0.3 x_chg[29] - 0.3 x_dis[29] + x_cost_ESS[29] = 0
 c_cost_OM_ESS[30]: - 0.3 x_chg[30] - 0.3 x_dis[30] + x_cost_ESS[30] = 0
 c_cost_OM_ESS[31]: - 0.3 x_chg[31] - 0.3 x_dis[31] + x_cost_ESS[31] = 0
 c_cost_OM_ESS[32]: - 0.3 x_chg[32] - 0.3 x_dis[32] + x_cost_ESS[32] = 0
 c_cost_OM_ESS[33]: - 0.3 x_chg[33] - 0.3 x_dis[33] + x_cost_ESS[33] = 0
 c_cost_OM_ESS[34]: - 0.3 x_chg[34] - 0.3 x_dis[34] + x_cost_ESS[34] = 0
 c_cost_OM_ESS[35]: - 0.3 x_chg[35] - 0.3 x_dis[35] + x_cost_ESS[35] = 0
 c_cost_OM_ESS[36]: - 0.3 x_chg[36] - 0.3 x_dis[36] + x_cost_ESS[36] = 0
 c_cost_OM_ESS[37]: - 0.3 x_chg[37] - 0.3 x_dis[37] + x_cost_ESS[37] = 0
 c_cost_OM_ESS[38]: - 0.3 x_chg[38] - 0.3 x_dis[38] + x_cost_ESS[38] = 0
 c_cost_OM_ESS[39]: - 0.3 x_chg[39] - 0.3 x_dis[39] + x_cost_ESS[39] = 0
 c_cost_OM_ESS[40]: - 0.3 x_chg[40] - 0.3 x_dis[40] + x_cost_ESS[40] = 0
 c_cost_OM_ESS[41]: - 0.3 x_chg[41] - 0.3 x_dis[41] + x_cost_ESS[41] = 0
 c_cost_OM_ESS[42]: - 0.3 x_chg[42] - 0.3 x_dis[42] + x_cost_ESS[42] = 0
 c_cost_OM_ESS[43]: - 0.3 x_chg[43] - 0.3 x_dis[43] + x_cost_ESS[43] = 0
 c_cost_OM_ESS[44]: - 0.3 x_chg[44] - 0.3 x_dis[44] + x_cost_ESS[44] = 0
 c_cost_OM_ESS[45]: - 0.3 x_chg[45] - 0.3 x_dis[45] + x_cost_ESS[45] = 0
 c_cost_OM_ESS[46]: - 0.3 x_chg[46] - 0.3 x_dis[46] + x_cost_ESS[46] = 0
 c_cost_OM_ESS[47]: - 0.3 x_chg[47] - 0.3 x_dis[47] + x_cost_ESS[47] = 0
 c_cost_OM_ESS[48]: - 0.3 x_chg[48] - 0.3 x_dis[48] + x_cost_ESS[48] = 0
 c_cost_OM_ESS[49]: - 0.3 x_chg[49] - 0.3 x_dis[49] + x_cost_ESS[49] = 0
 c_cost_OM_ESS[50]: - 0.3 x_chg[50] - 0.3 x_dis[50] + x_cost_ESS[50] = 0
 c_cost_OM_ESS[51]: - 0.3 x_chg[51] - 0.3 x_dis[51] + x_cost_ESS[51] = 0
 c_cost_OM_ESS[52]: - 0.3 x_chg[52] - 0.3 x_dis[52] + x_cost_ESS[52] = 0
 c_cost_OM_ESS[53]: - 0.3 x_chg[53] - 0.3 x_dis[53] + x_cost_ESS[53] = 0
 c_cost_OM_ESS[54]: - 0.3 x_chg[54] - 0.3 x_dis[54] + x_cost_ESS[54] = 0
 c_cost_OM_ESS[55]: - 0.3 x_chg[55] - 0.3 x_dis[55] + x_cost_ESS[55] = 0
 c_cost_OM_ESS[56]: - 0.3 x_chg[56] - 0.3 x_dis[56] + x_cost_ESS[56] = 0
 c_cost_OM_ESS[57]: - 0.3 x_chg[57] - 0.3 x_dis[57] + x_cost_ESS[57] = 0
 c_cost_OM_ESS[58]: - 0.3 x_chg[58] - 0.3 x_dis[58] + x_cost_ESS[58] = 0
 c_cost_OM_ESS[59]: - 0.3 x_chg[59] - 0.3 x_dis[59] + x_cost_ESS[59] = 0
 c_cost_OM_ESS[60]: - 0.3 x_chg[60] - 0.3 x_dis[60] + x_cost_ESS[60] = 0
 c_cost_OM_ESS[61]: - 0.3 x_chg[61] - 0.3 x_dis[61] + x_cost_ESS[61] = 0
 c_cost_OM_ESS[62]: - 0.3 x_chg[62] - 0.3 x_dis[62] + x_cost_ESS[62] = 0
 c_cost_OM_ESS[63]: - 0.3 x_chg[63] - 0.3 x_dis[63] + x_cost_ESS[63] = 0
 c_cost_OM_ESS[64]: - 0.3 x_chg[64] - 0.3 x_dis[64] + x_cost_ESS[64] = 0
 c_cost_OM_ESS[65]: - 0.3 x_chg[65] - 0.3 x_dis[65] + x_cost_ESS[65] = 0
 c_cost_OM_ESS[66]: - 0.3 x_chg[66] - 0.3 x_dis[66] + x_cost_ESS[66] = 0
 c_cost_OM_ESS[67]: - 0.3 x_chg[67] - 0.3 x_dis[67] + x_cost_ESS[67] = 0
 c_cost_OM_ESS[68]: - 0.3 x_chg[68] - 0.3 x_dis[68] + x_cost_ESS[68] = 0
 c_cost_OM_ESS[69]: - 0.3 x_chg[69] - 0.3 x_dis[69] + x_cost_ESS[69] = 0
 c_cost_OM_ESS[70]: - 0.3 x_chg[70] - 0.3 x_dis[70] + x_cost_ESS[70] = 0
 c_cost_OM_ESS[71]: - 0.3 x_chg[71] - 0.3 x_dis[71] + x_cost_ESS[71] = 0
 c_cost_OM_ESS[72]: - 0.3 x_chg[72] - 0.3 x_dis[72] + x_cost_ESS[72] = 0
 c_cost_OM_ESS[73]: - 0.3 x_chg[73] - 0.3 x_dis[73] + x_cost_ESS[73] = 0
 c_cost_OM_ESS[74]: - 0.3 x_chg[74] - 0.3 x_dis[74] + x_cost_ESS[74] = 0
 c_cost_OM_ESS[75]: - 0.3 x_chg[75] - 0.3 x_dis[75] + x_cost_ESS[75] = 0
 c_cost_OM_ESS[76]: - 0.3 x_chg[76] - 0.3 x_dis[76] + x_cost_ESS[76] = 0
 c_cost_OM_ESS[77]: - 0.3 x_chg[77] - 0.3 x_dis[77] + x_cost_ESS[77] = 0
 c_cost_OM_ESS[78]: - 0.3 x_chg[78] - 0.3 x_dis[78] + x_cost_ESS[78] = 0
 c_cost_OM_ESS[79]: - 0.3 x_chg[79] - 0.3 x_dis[79] + x_cost_ESS[79] = 0
 c_cost_OM_ESS[80]: - 0.3 x_chg[80] - 0.3 x_dis[80] + x_cost_ESS[80] = 0
 c_cost_OM_ESS[81]: - 0.3 x_chg[81] - 0.3 x_dis[81] + x_cost_ESS[81] = 0
 c_cost_OM_ESS[82]: - 0.3 x_chg[82] - 0.3 x_dis[82] + x_cost_ESS[82] = 0
 c_cost_OM_ESS[83]: - 0.3 x_chg[83] - 0.3 x_dis[83] + x_cost_ESS[83] = 0
 c_cost_OM_ESS[84]: - 0.3 x_chg[84] - 0.3 x_dis[84] + x_cost_ESS[84] = 0
 c_cost_OM_ESS[85]: - 0.3 x_chg[85] - 0.3 x_dis[85] + x_cost_ESS[85] = 0
 c_cost_OM_ESS[86]: - 0.3 x_chg[86] - 0.3 x_dis[86] + x_cost_ESS[86] = 0
 c_cost_OM_ESS[87]: - 0.3 x_chg[87] - 0.3 x_dis[87] + x_cost_ESS[87] = 0
 c_cost_OM_ESS[88]: - 0.3 x_chg[88] - 0.3 x_dis[88] + x_cost_ESS[88] = 0
 c_cost_OM_ESS[89]: - 0.3 x_chg[89] - 0.3 x_dis[89] + x_cost_ESS[89] = 0
 c_cost_OM_ESS[90]: - 0.3 x_chg[90] - 0.3 x_dis[90] + x_cost_ESS[90] = 0
 c_cost_OM_ESS[91]: - 0.3 x_chg[91] - 0.3 x_dis[91] + x_cost_ESS[91] = 0
 c_cost_OM_ESS[92]: - 0.3 x_chg[92] - 0.3 x_dis[92] + x_cost_ESS[92] = 0
 c_cost_OM_ESS[93]: - 0.3 x_chg[93] - 0.3 x_dis[93] + x_cost_ESS[93] = 0
 c_cost_OM_ESS[94]: - 0.3 x_chg[94] - 0.3 x_dis[94] + x_cost_ESS[94] = 0
 c_cost_OM_ESS[95]: - 0.3 x_chg[95] - 0.3 x_dis[95] + x_cost_ESS[95] = 0
 c_reserve_pos_DG[0]: - x_pos[0] + y_pos[0] <= 0
 c_reserve_pos_DG[1]: - x_pos[1] + y_pos[1] <= 0
 c_reserve_pos_DG[2]: - x_pos[2] + y_pos[2] <= 0
 c_reserve_pos_DG[3]: - x_pos[3] + y_pos[3] <= 0
 c_reserve_pos_DG[4]: - x_pos[4] + y_pos[4] <= 0
 c_reserve_pos_DG[5]: - x_pos[5] + y_pos[5] <= 0
 c_reserve_pos_DG[6]: - x_pos[6] + y_pos[6] <= 0
 c_reserve_pos_DG[7]: - x_pos[7] + y_pos[7] <= 0
 c_reserve_pos_DG[8]: - x_pos[8] + y_pos[8] <= 0
 c_reserve_pos_DG[9]: - x_pos[9] + y_pos[9] <= 0
 c_reserve_pos_DG[10]: - x_pos[10] + y_pos[10] <= 0
 c_reserve_pos_DG[11]: - x_pos[11] + y_pos[11] <= 0
 c_reserve_pos_DG[12]: - x_pos[12] + y_pos[12] <= 0
 c_reserve_pos_DG[13]: - x_pos[13] + y_pos[13] <= 0
 c_reserve_pos_DG[14]: - x_pos[14] + y_pos[14] <= 0
 c_reserve_pos_DG[15]: - x_pos[15] + y_pos[15] <= 0
 c_reserve_pos_DG[16]: - x_pos[16] + y_pos[16] <= 0
 c_reserve_pos_DG[17]: - x_pos[17] + y_pos[17] <= 0
 c_reserve_pos_DG[18]: - x_pos[18] + y_pos[18] <= 0
 c_reserve_pos_DG[19]: - x_pos[19] + y_pos[19] <= 0
 c_reserve_pos_DG[20]: - x_pos[20] + y_pos[20] <= 0
 c_reserve_pos_DG[21]: - x_pos[21] + y_pos[21] <= 0
 c_reserve_pos_DG[22]: - x_pos[22] + y_pos[22] <= 0
 c_reserve_pos_DG[23]: - x_pos[23] + y_pos[23] <= 0
 c_reserve_pos_DG[24]: - x_pos[24] + y_pos[24] <= 0
 c_reserve_pos_DG[25]: - x_pos[25] + y_pos[25] <= 0
 c_reserve_pos_DG[26]: - x_pos[26] + y_pos[26] <= 0
 c_reserve_pos_DG[27]: - x_pos[27] + y_pos[27] <= 0
 c_reserve_pos_DG[28]: - x_pos[28] + y_pos[28] <= 0
 c_reserve_pos_DG[29]: - x_pos[29] + y_pos[29] <= 0
 c_reserve_pos_DG[30]: - x_pos[30] + y_pos[30] <= 0
 c_reserve_pos_DG[31]: - x_pos[31] + y_pos[31] <= 0
 c_reserve_pos_DG[32]: - x_pos[32] + y_pos[32] <= 0
 c_reserve_pos_DG[33]: - x_pos[33] + y_pos[33] <= 0
 c_reserve_pos_DG[34]: - x_pos[34] + y_pos[34] <= 0
 c_reserve_pos_DG[35]: - x_pos[35] + y_pos[35] <= 0
 c_reserve_pos_DG[36]: - x_pos[36] + y_pos[36] <= 0
 c_reserve_pos_DG[37]: - x_pos[37] + y_pos[37] <= 0
 c_reserve_pos_DG[38]: - x_pos[38] + y_pos[38] <= 0
 c_reserve_pos_DG[39]: - x_pos[39] + y_pos[39] <= 0
 c_reserve_pos_DG[40]: - x_pos[40] + y_pos[40] <= 0
 c_reserve_pos_DG[41]: - x_pos[41] + y_pos[41] <= 0
 c_reserve_pos_DG[42]: - x_pos[42] + y_pos[42] <= 0
 c_reserve_pos_DG[43]: - x_pos[43] + y_pos[43] <= 0
 c_reserve_pos_DG[44]: - x_pos[44] + y_pos[44] <= 0
 c_reserve_pos_DG[45]: - x_pos[45] + y_pos[45] <= 0
 c_reserve_pos_DG[46]: - x_pos[46] + y_pos[46] <= 0
 c_reserve_pos_DG[47]: - x_pos[47] + y_pos[47] <= 0
 c_reserve_pos_DG[48]: - x_pos[48] + y_pos[48] <= 0
 c_reserve_pos_DG[49]: - x_pos[49] + y_pos[49] <= 0
 c_reserve_pos_DG[50]: - x_pos[50] + y_pos[50] <= 0
 c_reserve_pos_DG[51]: - x_pos[51] + y_pos[51] <= 0
 c_reserve_pos_DG[52]: - x_pos[52] + y_pos[52] <= 0
 c_reserve_pos_DG[53]: - x_pos[53] + y_pos[53] <= 0
 c_reserve_pos_DG[54]: - x_pos[54] + y_pos[54] <= 0
 c_reserve_pos_DG[55]: - x_pos[55] + y_pos[55] <= 0
 c_reserve_pos_DG[56]: - x_pos[56] + y_pos[56] <= 0
 c_reserve_pos_DG[57]: - x_pos[57] + y_pos[57] <= 0
 c_reserve_pos_DG[58]: - x_pos[58] + y_pos[58] <= 0
 c_reserve_pos_DG[59]: - x_pos[59] + y_pos[59] <= 0
 c_reserve_pos_DG[60]: - x_pos[60] + y_pos[60] <= 0
 c_reserve_pos_DG[61]: - x_pos[61] + y_pos[61] <= 0
 c_reserve_pos_DG[62]: - x_pos[62] + y_pos[62] <= 0
 c_reserve_pos_DG[63]: - x_pos[63] + y_pos[63] <= 0
 c_reserve_pos_DG[64]: - x_pos[64] + y_pos[64] <= 0
 c_reserve_pos_DG[65]: - x_pos[65] + y_pos[65] <= 0
 c_reserve_pos_DG[66]: - x_pos[66] + y_pos[66] <= 0
 c_reserve_pos_DG[67]: - x_pos[67] + y_pos[67] <= 0
 c_reserve_pos_DG[68]: - x_pos[68] + y_pos[68] <= 0
 c_reserve_pos_DG[69]: - x_pos[69] + y_pos[69] <= 0
 c_reserve_pos_DG[70]: - x_pos[70] + y_pos[70] <= 0
 c_reserve_pos_DG[71]: - x_pos[71] + y_pos[71] <= 0
 c_reserve_pos_DG[72]: - x_pos[72] + y_pos[72] <= 0
 c_reserve_pos_DG[73]: - x_pos[73] + y_pos[73] <= 0
 c_reserve_pos_DG[74]: - x_pos[74] + y_pos[74] <= 0
 c_reserve_pos_DG[75]: - x_pos[75] + y_pos[75] <= 0
 c_reserve_pos_DG[76]: - x_pos[76] + y_pos[76] <= 0
 c_reserve_pos_DG[77]: - x_pos[77] + y_pos[77] <= 0
 c_reserve_pos_DG[78]: - x_pos[78] + y_pos[78] <= 0
 c_reserve_pos_DG[79]: - x_pos[79] + y_pos[79] <= 0
 c_reserve_pos_DG[80]: - x_pos[80] + y_pos[80] <= 0
 c_reserve_pos_DG[81]: - x_pos[81] + y_pos[81] <= 0
 c_reserve_pos_DG[82]: - x_pos[82] + y_pos[82] <= 0
 c_reserve_pos_DG[83]: - x_pos[83] + y_pos[83] <= 0
 c_reserve_pos_DG[84]: - x_pos[84] + y_pos[84] <= 0
 c_reserve_pos_DG[85]: - x_pos[85] + y_pos[85] <= 0
 c_reserve_pos_DG[86]: - x_pos[86] + y_pos[86] <= 0
 c_reserve_pos_DG[87]: - x_pos[87] + y_pos[87] <= 0
 c_reserve_pos_DG[88]: - x_pos[88] + y_pos[88] <= 0
 c_reserve_pos_DG[89]: - x_pos[89] + y_pos[89] <= 0
 c_reserve_pos_DG[90]: - x_pos[90] + y_pos[90] <= 0
 c_reserve_pos_DG[91]: - x_pos[91] + y_pos[91] <= 0
 c_reserve_pos_DG[92]: - x_pos[92] + y_pos[92] <= 0
 c_reserve_pos_DG[93]: - x_pos[93] + y_pos[93] <= 0
 c_reserve_pos_DG[94]: - x_pos[94] + y_pos[94] <= 0
 c_reserve_pos_DG[95]: - x_pos[95] + y_pos[95] <= 0
 c_reserve_neg_DG[0]: - x_neg[0] + y_neg[0] <= 0
 c_reserve_neg_DG[1]: - x_neg[1] + y_neg[1] <= 0
 c_reserve_neg_DG[2]: - x_neg[2] + y_neg[2] <= 0
 c_reserve_neg_DG[3]: - x_neg[3] + y_neg[3] <= 0
 c_reserve_neg_DG[4]: - x_neg[4] + y_neg[4] <= 0
 c_reserve_neg_DG[5]: - x_neg[5] + y_neg[5] <= 0
 c_reserve_neg_DG[6]: - x_neg[6] + y_neg[6] <= 0
 c_reserve_neg_DG[7]: - x_neg[7] + y_neg[7] <= 0
 c_reserve_neg_DG[8]: - x_neg[8] + y_neg[8] <= 0
 c_reserve_neg_DG[9]: - x_neg[9] + y_neg[9] <= 0
 c_reserve_neg_DG[10]: - x_neg[10] + y_neg[10] <= 0
 c_reserve_neg_DG[11]: - x_neg[11] + y_neg[11] <= 0
 c_reserve_neg_DG[12]: - x_neg[12] + y_neg[12] <= 0
 c_reserve_neg_DG[13]: - x_neg[13] + y_neg[13] <= 0
 c_reserve_neg_DG[14]: - x_neg[14] + y_neg[14] <= 0
 c_reserve_neg_DG[15]: - x_neg[15] + y_neg[15] <= 0
 c_reserve_neg_DG[16]: - x_neg[16] + y_neg[16] <= 0
 c_reserve_neg_DG[17]: - x_neg[17] + y_neg[17] <= 0
 c_reserve_neg_DG[18]: - x_neg[18] + y_neg[18] <= 0
 c_reserve_neg_DG[19]: - x_neg[19] + y_neg[19] <= 0
 c_reserve_neg_DG[20]: - x_neg[20] + y_neg[20] <= 0
 c_reserve_neg_DG[21]: - x_neg[21] + y_neg[21] <= 0
 c_reserve_neg_DG[22]: - x_neg[22] + y_neg[22] <= 0
 c_reserve_neg_DG[23]: - x_neg[23] + y_neg[23] <= 0
 c_reserve_neg_DG[24]: - x_neg[24] + y_neg[24] <= 0
 c_reserve_neg_DG[25]: - x_neg[25] + y_neg[25] <= 0
 c_reserve_neg_DG[26]: - x_neg[26] + y_neg[26] <= 0
 c_reserve_neg_DG[27]: - x_neg[27] + y_neg[27] <= 0
 c_reserve_neg_DG[28]: - x_neg[28] + y_neg[28] <= 0
 c_reserve_neg_DG[29]: - x_neg[29] + y_neg[29] <= 0
 c_reserve_neg_DG[30]: - x_neg[30] + y_neg[30] <= 0
 c_reserve_neg_DG[31]: - x_neg[31] + y_neg[31] <= 0
 c_reserve_neg_DG[32]: - x_neg[32] + y_neg[32] <= 0
 c_reserve_neg_DG[33]: - x_neg[33] + y_neg[33] <= 0
 c_reserve_neg_DG[34]: - x_neg[34] + y_neg[34] <= 0
 c_reserve_neg_DG[35]: - x_neg[35] + y_neg[35] <= 0
 c_reserve_neg_DG[36]: - x_neg[36] + y_neg[36] <= 0
 c_reserve_neg_DG[37]: - x_neg[37] + y_neg[37] <= 0
 c_reserve_neg_DG[38]: - x_neg[38] + y_neg[38] <= 0
 c_reserve_neg_DG[39]: - x_neg[39] + y_neg[39] <= 0
 c_reserve_neg_DG[40]: - x_neg[40] + y_neg[40] <= 0
 c_reserve_neg_DG[41]: - x_neg[41] + y_neg[41] <= 0
 c_reserve_neg_DG[42]: - x_neg[42] + y_neg[42] <= 0
 c_reserve_neg_DG[43]: - x_neg[43] + y_neg[43] <= 0
 c_reserve_neg_DG[44]: - x_neg[44] + y_neg[44] <= 0
 c_reserve_neg_DG[45]: - x_neg[45] + y_neg[45] <= 0
 c_reserve_neg_DG[46]: - x_neg[46] + y_neg[46] <= 0
 c_reserve_neg_DG[47]: - x_neg[47] + y_neg[47] <= 0
 c_reserve_neg_DG[48]: - x_neg[48] + y_neg[48] <= 0
 c_reserve_neg_DG[49]: - x_neg[49] + y_neg[49] <= 0
 c_reserve_neg_DG[50]: - x_neg[50] + y_neg[50] <= 0
 c_reserve_neg_DG[51]: - x_neg[51] + y_neg[51] <= 0
 c_reserve_neg_DG[52]: - x_neg[52] + y_neg[52] <= 0
 c_reserve_neg_DG[53]: - x_neg[53] + y_neg[53] <= 0
 c_reserve_neg_DG[54]: - x_neg[54] + y_neg[54] <= 0
 c_reserve_neg_DG[55]: - x_neg[55] + y_neg[55] <= 0
 c_reserve_neg_DG[56]: - x_neg[56] + y_neg[56] <= 0
 c_reserve_neg_DG[57]: - x_neg[57] + y_neg[57] <= 0
 c_reserve_neg_DG[58]: - x_neg[58] + y_neg[58] <= 0
 c_reserve_neg_DG[59]: - x_neg[59] + y_neg[59] <= 0
 c_reserve_neg_DG[60]: - x_neg[60] + y_neg[60] <= 0
 c_reserve_neg_DG[61]: - x_neg[61] + y_neg[61] <= 0
 c_reserve_neg_DG[62]: - x_neg[62] + y_neg[62] <= 0
 c_reserve_neg_DG[63]: - x_neg[63] + y_neg[63] <= 0
 c_reserve_neg_DG[64]: - x_neg[64] + y_neg[64] <= 0
 c_reserve_neg_DG[65]: - x_neg[65] + y_neg[65] <= 0
 c_reserve_neg_DG[66]: - x_neg[66] + y_neg[66] <= 0
 c_reserve_neg_DG[67]: - x_neg[67] + y_neg[67] <= 0
 c_reserve_neg_DG[68]: - x_neg[68] + y_neg[68] <= 0
 c_reserve_neg_DG[69]: - x_neg[69] + y_neg[69] <= 0
 c_reserve_neg_DG[70]: - x_neg[70] + y_neg[70] <= 0
 c_reserve_neg_DG[71]: - x_neg[71] + y_neg[71] <= 0
 c_reserve_neg_DG[72]: - x_neg[72] + y_neg[72] <= 0
 c_reserve_neg_DG[73]: - x_neg[73] + y_neg[73] <= 0
 c_reserve_neg_DG[74]: - x_neg[74] + y_neg[74] <= 0
 c_reserve_neg_DG[75]: - x_neg[75] + y_neg[75] <= 0
 c_reserve_neg_DG[76]: - x_neg[76] + y_neg[76] <= 0
 c_reserve_neg_DG[77]: - x_neg[77] + y_neg[77] <= 0
 c_reserve_neg_DG[78]: - x_neg[78] + y_neg[78] <= 0
 c_reserve_neg_DG[79]: - x_neg[79] + y_neg[79] <= 0
 c_reserve_neg_DG[80]: - x_neg[80] + y_neg[80] <= 0
 c_reserve_neg_DG[81]: - x_neg[81] + y_neg[81] <= 0
 c_reserve_neg_DG[82]: - x_neg[82] + y_neg[82] <= 0
 c_reserve_neg_DG[83]: - x_neg[83] + y_neg[83] <= 0
 c_reserve_neg_DG[84]: - x_neg[84] + y_neg[84] <= 0
 c_reserve_neg_DG[85]: - x_neg[85] + y_neg[85] <= 0
 c_reserve_neg_DG[86]: - x_neg[86] + y_neg[86] <= 0
 c_reserve_neg_DG[87]: - x_neg[87] + y_neg[87] <= 0
 c_reserve_neg_DG[88]: - x_neg[88] + y_neg[88] <= 0
 c_reserve_neg_DG[89]: - x_neg[89] + y_neg[89] <= 0
 c_reserve_neg_DG[90]: - x_neg[90] + y_neg[90] <= 0
 c_reserve_neg_DG[91]: - x_neg[91] + y_neg[91] <= 0
 c_reserve_neg_DG[92]: - x_neg[92] + y_neg[92] <= 0
 c_reserve_neg_DG[93]: - x_neg[93] + y_neg[93] <= 0
 c_reserve_neg_DG[94]: - x_neg[94] + y_neg[94] <= 0
 c_reserve_neg_DG[95]: - x_neg[95] + y_neg[95] <= 0
 c_discharge_re[0]: - 250 y_b[0] + y_dis[0] <= 0
 c_discharge_re[1]: - 250 y_b[1] + y_dis[1] <= 0
 c_discharge_re[2]: - 250 y_b[2] + y_dis[2] <= 0
 c_discharge_re[3]: - 250 y_b[3] + y_dis[3] <= 0
 c_discharge_re[4]: - 250 y_b[4] + y_dis[4] <= 0
 c_discharge_re[5]: - 250 y_b[5] + y_dis[5] <= 0
 c_discharge_re[6]: - 250 y_b[6] + y_dis[6] <= 0
 c_discharge_re[7]: - 250 y_b[7] + y_dis[7] <= 0
 c_discharge_re[8]: - 250 y_b[8] + y_dis[8] <= 0
 c_discharge_re[9]: - 250 y_b[9] + y_dis[9] <= 0
 c_discharge_re[10]: - 250 y_b[10] + y_dis[10] <= 0
 c_discharge_re[11]: - 250 y_b[11] + y_dis[11] <= 0
 c_discharge_re[12]: - 250 y_b[12] + y_dis[12] <= 0
 c_discharge_re[13]: - 250 y_b[13] + y_dis[13] <= 0
 c_discharge_re[14]: - 250 y_b[14] + y_dis[14] <= 0
 c_discharge_re[15]: - 250 y_b[15] + y_dis[15] <= 0
 c_discharge_re[16]: - 250 y_b[16] + y_dis[16] <= 0
 c_discharge_re[17]: - 250 y_b[17] + y_dis[17] <= 0
 c_discharge_re[18]: - 250 y_b[18] + y_dis[18] <= 0
 c_discharge_re[19]: - 250 y_b[19] + y_dis[19] <= 0
 c_discharge_re[20]: - 250 y_b[20] + y_dis[20] <= 0
 c_discharge_re[21]: - 250 y_b[21] + y_dis[21] <= 0
 c_discharge_re[22]: - 250 y_b[22] + y_dis[22] <= 0
 c_discharge_re[23]: - 250 y_b[23] + y_dis[23] <= 0
 c_discharge_re[24]: - 250 y_b[24] + y_dis[24] <= 0
 c_discharge_re[25]: - 250 y_b[25] + y_dis[25] <= 0
 c_discharge_re[26]: - 250 y_b[26] + y_dis[26] <= 0
 c_discharge_re[27]: - 250 y_b[27] + y_dis[27] <= 0
 c_discharge_re[28]: - 250 y_b[28] + y_dis[28] <= 0
 c_discharge_re[29]: - 250 y_b[29] + y_dis[29] <= 0
 c_discharge_re[30]: - 250 y_b[30] + y_dis[30] <= 0
 c_discharge_re[31]: - 250 y_b[31] + y_dis[31] <= 0
 c_discharge_re[32]: - 250 y_b[32] + y_dis[32] <= 0
 c_discharge_re[33]: - 250 y_b[33] + y_dis[33] <= 0
 c_discharge_re[34]: - 250 y_b[34] + y_dis[34] <= 0
 c_discharge_re[35]: - 250 y_b[35] + y_dis[35] <= 0
 c_discharge_re[36]: - 250 y_b[36] + y_dis[36] <= 0
 c_discharge_re[37]: - 250 y_b[37] + y_dis[37] <= 0
 c_discharge_re[38]: - 250 y_b[38] + y_dis[38] <= 0
 c_discharge_re[39]: - 250 y_b[39] + y_dis[39] <= 0
 c_discharge_re[40]: - 250 y_b[40] + y_dis[40] <= 0
 c_discharge_re[41]: - 250 y_b[41] + y_dis[41] <= 0
 c_discharge_re[42]: - 250 y_b[42] + y_dis[42] <= 0
 c_discharge_re[43]: - 250 y_b[43] + y_dis[43] <= 0
 c_discharge_re[44]: - 250 y_b[44] + y_dis[44] <= 0
 c_discharge_re[45]: - 250 y_b[45] + y_dis[45] <= 0
 c_discharge_re[46]: - 250 y_b[46] + y_dis[46] <= 0
 c_discharge_re[47]: - 250 y_b[47] + y_dis[47] <= 0
 c_discharge_re[48]: - 250 y_b[48] + y_dis[48] <= 0
 c_discharge_re[49]: - 250 y_b[49] + y_dis[49] <= 0
 c_discharge_re[50]: - 250 y_b[50] + y_dis[50] <= 0
 c_discharge_re[51]: - 250 y_b[51] + y_dis[51] <= 0
 c_discharge_re[52]: - 250 y_b[52] + y_dis[52] <= 0
 c_discharge_re[53]: - 250 y_b[53] + y_dis[53] <= 0
 c_discharge_re[54]: - 250 y_b[54] + y_dis[54] <= 0
 c_discharge_re[55]: - 250 y_b[55] + y_dis[55] <= 0
 c_discharge_re[56]: - 250 y_b[56] + y_dis[56] <= 0
 c_discharge_re[57]: - 250 y_b[57] + y_dis[57] <= 0
 c_discharge_re[58]: - 250 y_b[58] + y_dis[58] <= 0
 c_discharge_re[59]: - 250 y_b[59] + y_dis[59] <= 0
 c_discharge_re[60]: - 250 y_b[60] + y_dis[60] <= 0
 c_discharge_re[61]: - 250 y_b[61] + y_dis[61] <= 0
 c_discharge_re[62]: - 250 y_b[62] + y_dis[62] <= 0
 c_discharge_re[63]: - 250 y_b[63] + y_dis[63] <= 0
 c_discharge_re[64]: - 250 y_b[64] + y_dis[64] <= 0
 c_discharge_re[65]: - 250 y_b[65] + y_dis[65] <= 0
 c_discharge_re[66]: - 250 y_b[66] + y_dis[66] <= 0
 c_discharge_re[67]: - 250 y_b[67] + y_dis[67] <= 0
 c_discharge_re[68]: - 250 y_b[68] + y_dis[68] <= 0
 c_discharge_re[69]: - 250 y_b[69] + y_dis[69] <= 0
 c_discharge_re[70]: - 250 y_b[70] + y_dis[70] <= 0
 c_discharge_re[71]: - 250 y_b[71] + y_dis[71] <= 0
 c_discharge_re[72]: - 250 y_b[72] + y_dis[72] <= 0
 c_discharge_re[73]: - 250 y_b[73] + y_dis[73] <= 0
 c_discharge_re[74]: - 250 y_b[74] + y_dis[74] <= 0
 c_discharge_re[75]: - 250 y_b[75] + y_dis[75] <= 0
 c_discharge_re[76]: - 250 y_b[76] + y_dis[76] <= 0
 c_discharge_re[77]: - 250 y_b[77] + y_dis[77] <= 0
 c_discharge_re[78]: - 250 y_b[78] + y_dis[78] <= 0
 c_discharge_re[79]: - 250 y_b[79] + y_dis[79] <= 0
 c_discharge_re[80]: - 250 y_b[80] + y_dis[80] <= 0
 c_discharge_re[81]: - 250 y_b[81] + y_dis[81] <= 0
 c_discharge_re[82]: - 250 y_b[82] + y_dis[82] <= 0
 c_discharge_re[83]: - 250 y_b[83] + y_dis[83] <= 0
 c_discharge_re[84]: - 250 y_b[84] + y_dis[84] <= 0
 c_discharge_re[85]: - 250 y_b[85] + y_dis[85] <= 0
 c_discharge_re[86]: - 250 y_b[86] + y_dis[86] <= 0
 c_discharge_re[87]: - 250 y_b[87] + y_dis[87] <= 0
 c_discharge_re[88]: - 250 y_b[88] + y_dis[88] <= 0
 c_discharge_re[89]: - 250 y_b[89] + y_dis[89] <= 0
 c_discharge_re[90]: - 250 y_b[90] + y_dis[90] <= 0
 c_discharge_re[91]: - 250 y_b[91] + y_dis[91] <= 0
 c_discharge_re[92]: - 250 y_b[92] + y_dis[92] <= 0
 c_discharge_re[93]: - 250 y_b[93] + y_dis[93] <= 0
 c_discharge_re[94]: - 250 y_b[94] + y_dis[94] <= 0
 c_discharge_re[95]: - 250 y_b[95] + y_dis[95] <= 0
 c_charge_re[0]: 250 y_b[0] + y_chg[0] <= 250
 c_charge_re[1]: 250 y_b[1] + y_chg[1] <= 250
 c_charge_re[2]: 250 y_b[2] + y_chg[2] <= 250
 c_charge_re[3]: 250 y_b[3] + y_chg[3] <= 250
 c_charge_re[4]: 250 y_b[4] + y_chg[4] <= 250
 c_charge_re[5]: 250 y_b[5] + y_chg[5] <= 250
 c_charge_re[6]: 250 y_b[6] + y_chg[6] <= 250
 c_charge_re[7]: 250 y_b[7] + y_chg[7] <= 250
 c_charge_re[8]: 250 y_b[8] + y_chg[8] <= 250
 c_charge_re[9]: 250 y_b[9] + y_chg[9] <= 250
 c_charge_re[10]: 250 y_b[10] + y_chg[10] <= 250
 c_charge_re[11]: 250 y_b[11] + y_chg[11] <= 250
 c_charge_re[12]: 250 y_b[12] + y_chg[12] <= 250
 c_charge_re[13]: 250 y_b[13] + y_chg[13] <= 250
 c_charge_re[14]: 250 y_b[14] + y_chg[14] <= 250
 c_charge_re[15]: 250 y_b[15] + y_chg[15] <= 250
 c_charge_re[16]: 250 y_b[16] + y_chg[16] <= 250
 c_charge_re[17]: 250 y_b[17] + y_chg[17] <= 250
 c_charge_re[18]: 250 y_b[18] + y_chg[18] <= 250
 c_charge_re[19]: 250 y_b[19] + y_chg[19] <= 250
 c_charge_re[20]: 250 y_b[20] + y_chg[20] <= 250
 c_charge_re[21]: 250 y_b[21] + y_chg[21] <= 250
 c_charge_re[22]: 250 y_b[22] + y_chg[22] <= 250
 c_charge_re[23]: 250 y_b[23] + y_chg[23] <= 250
 c_charge_re[24]: 250 y_b[24] + y_chg[24] <= 250
 c_charge_re[25]: 250 y_b[25] + y_chg[25] <= 250
 c_charge_re[26]: 250 y_b[26] + y_chg[26] <= 250
 c_charge_re[27]: 250 y_b[27] + y_chg[27] <= 250
 c_charge_re[28]: 250 y_b[28] + y_chg[28] <= 250
 c_charge_re[29]: 250 y_b[29] + y_chg[29] <= 250
 c_charge_re[30]: 250 y_b[30] + y_chg[30] <= 250
 c_charge_re[31]: 250 y_b[31] + y_chg[31] <= 250
 c_charge_re[32]: 250 y_b[32] + y_chg[32] <= 250
 c_charge_re[33]: 250 y_b[33] + y_chg[33] <= 250
 c_charge_re[34]: 250 y_b[34] + y_chg[34] <= 250
 c_charge_re[35]: 250 y_b[35] + y_chg[35] <= 250
 c_charge_re[36]: 250 y_b[36] + y_chg[36] <= 250
 c_charge_re[37]: 250 y_b[37] + y_chg[37] <= 250
 c_charge_re[38]: 250 y_b[38] + y_chg[38] <= 250
 c_charge_re[39]: 250 y_b[39] + y_chg[39] <= 250
 c_charge_re[40]: 250 y_b[40] + y_chg[40] <= 250
 c_charge_re[41]: 250 y_b[41] + y_chg[41] <= 250
 c_charge_re[42]: 250 y_b[42] + y_chg[42] <= 250
 c_charge_re[43]: 250 y_b[43] + y_chg[43] <= 250
 c_charge_re[44]: 250 y_b[44] + y_chg[44] <= 250
 c_charge_re[45]: 250 y_b[45] + y_chg[45] <= 250
 c_charge_re[46]: 250 y_b[46] + y_chg[46] <= 250
 c_charge_re[47]: 250 y_b[47] + y_chg[47] <= 250
 c_charge_re[48]: 250 y_b[48] + y_chg[48] <= 250
 c_charge_re[49]: 250 y_b[49] + y_chg[49] <= 250
 c_charge_re[50]: 250 y_b[50] + y_chg[50] <= 250
 c_charge_re[51]: 250 y_b[51] + y_chg[51] <= 250
 c_charge_re[52]: 250 y_b[52] + y_chg[52] <= 250
 c_charge_re[53]: 250 y_b[53] + y_chg[53] <= 250
 c_charge_re[54]: 250 y_b[54] + y_chg[54] <= 250
 c_charge_re[55]: 250 y_b[55] + y_chg[55] <= 250
 c_charge_re[56]: 250 y_b[56] + y_chg[56] <= 250
 c_charge_re[57]: 250 y_b[57] + y_chg[57] <= 250
 c_charge_re[58]: 250 y_b[58] + y_chg[58] <= 250
 c_charge_re[59]: 250 y_b[59] + y_chg[59] <= 250
 c_charge_re[60]: 250 y_b[60] + y_chg[60] <= 250
 c_charge_re[61]: 250 y_b[61] + y_chg[61] <= 250
 c_charge_re[62]: 250 y_b[62] + y_chg[62] <= 250
 c_charge_re[63]: 250 y_b[63] + y_chg[63] <= 250
 c_charge_re[64]: 250 y_b[64] + y_chg[64] <= 250
 c_charge_re[65]: 250 y_b[65] + y_chg[65] <= 250
 c_charge_re[66]: 250 y_b[66] + y_chg[66] <= 250
 c_charge_re[67]: 250 y_b[67] + y_chg[67] <= 250
 c_charge_re[68]: 250 y_b[68] + y_chg[68] <= 250
 c_charge_re[69]: 250 y_b[69] + y_chg[69] <= 250
 c_charge_re[70]: 250 y_b[70] + y_chg[70] <= 250
 c_charge_re[71]: 250 y_b[71] + y_chg[71] <= 250
 c_charge_re[72]: 250 y_b[72] + y_chg[72] <= 250
 c_charge_re[73]: 250 y_b[73] + y_chg[73] <= 250
 c_charge_re[74]: 250 y_b[74] + y_chg[74] <= 250
 c_charge_re[75]: 250 y_b[75] + y_chg[75] <= 250
 c_charge_re[76]: 250 y_b[76] + y_chg[76] <= 250
 c_charge_re[77]: 250 y_b[77] + y_chg[77] <= 250
 c_charge_re[78]: 250 y_b[78] + y_chg[78] <= 250
 c_charge_re[79]: 250 y_b[79] + y_chg[79] <= 250
 c_charge_re[80]: 250 y_b[80] + y_chg[80] <= 250
 c_charge_re[81]: 250 y_b[81] + y_chg[81] <= 250
 c_charge_re[82]: 250 y_b[82] + y_chg[82] <= 250
 c_charge_re[83]: 250 y_b[83] + y_chg[83] <= 250
 c_charge_re[84]: 250 y_b[84] + y_chg[84] <= 250
 c_charge_re[85]: 250 y_b[85] + y_chg[85] <= 250
 c_charge_re[86]: 250 y_b[86] + y_chg[86] <= 250
 c_charge_re[87]: 250 y_b[87] + y_chg[87] <= 250
 c_charge_re[88]: 250 y_b[88] + y_chg[88] <= 250
 c_charge_re[89]: 250 y_b[89] + y_chg[89] <= 250
 c_charge_re[90]: 250 y_b[90] + y_chg[90] <= 250
 c_charge_re[91]: 250 y_b[91] + y_chg[91] <= 250
 c_charge_re[92]: 250 y_b[92] + y_chg[92] <= 250
 c_charge_re[93]: 250 y_b[93] + y_chg[93] <= 250
 c_charge_re[94]: 250 y_b[94] + y_chg[94] <= 250
 c_charge_re[95]: 250 y_b[95] + y_chg[95] <= 250
 c_min_S[0]: - y_S[0] <= -113.4
 c_min_S[1]: - y_S[1] <= -113.4
 c_min_S[2]: - y_S[2] <= -113.4
 c_min_S[3]: - y_S[3] <= -113.4
 c_min_S[4]: - y_S[4] <= -113.4
 c_min_S[5]: - y_S[5] <= -113.4
 c_min_S[6]: - y_S[6] <= -113.4
 c_min_S[7]: - y_S[7] <= -113.4
 c_min_S[8]: - y_S[8] <= -113.4
 c_min_S[9]: - y_S[9] <= -113.4
 c_min_S[10]: - y_S[10] <= -113.4
 c_min_S[11]: - y_S[11] <= -113.4
 c_min_S[12]: - y_S[12] <= -113.4
 c_min_S[13]: - y_S[13] <= -113.4
 c_min_S[14]: - y_S[14] <= -113.4
 c_min_S[15]: - y_S[15] <= -113.4
 c_min_S[16]: - y_S[16] <= -113.4
 c_min_S[17]: - y_S[17] <= -113.4
 c_min_S[18]: - y_S[18] <= -113.4
 c_min_S[19]: - y_S[19] <= -113.4
 c_min_S[20]: - y_S[20] <= -113.4
 c_min_S[21]: - y_S[21] <= -113.4
 c_min_S[22]: - y_S[22] <= -113.4
 c_min_S[23]: - y_S[23] <= -113.4
 c_min_S[24]: - y_S[24] <= -113.4
 c_min_S[25]: - y_S[25] <= -113.4
 c_min_S[26]: - y_S[26] <= -113.4
 c_min_S[27]: - y_S[27] <= -113.4
 c_min_S[28]: - y_S[28] <= -113.4
 c_min_S[29]: - y_S[29] <= -113.4
 c_min_S[30]: - y_S[30] <= -113.4
 c_min_S[31]: - y_S[31] <= -113.4
 c_min_S[32]: - y_S[32] <= -113.4
 c_min_S[33]: - y_S[33] <= -113.4
 c_min_S[34]: - y_S[34] <= -113.4
 c_min_S[35]: - y_S[35] <= -113.4
 c_min_S[36]: - y_S[36] <= -113.4
 c_min_S[37]: - y_S[37] <= -113.4
 c_min_S[38]: - y_S[38] <= -113.4
 c_min_S[39]: - y_S[39] <= -113.4
 c_min_S[40]: - y_S[40] <= -113.4
 c_min_S[41]: - y_S[41] <= -113.4
 c_min_S[42]: - y_S[42] <= -113.4
 c_min_S[43]: - y_S[43] <= -113.4
 c_min_S[44]: - y_S[44] <= -113.4
 c_min_S[45]: - y_S[45] <= -113.4
 c_min_S[46]: - y_S[46] <= -113.4
 c_min_S[47]: - y_S[47] <= -113.4
 c_min_S[48]: - y_S[48] <= -113.4
 c_min_S[49]: - y_S[49] <= -113.4
 c_min_S[50]: - y_S[50] <= -113.4
 c_min_S[51]: - y_S[51] <= -113.4
 c_min_S[52]: - y_S[52] <= -113.4
 c_min_S[53]: - y_S[53] <= -113.4
 c_min_S[54]: - y_S[54] <= -113.4
 c_min_S[55]: - y_S[55] <= -113.4
 c_min_S[56]: - y_S[56] <= -113.4
 c_min_S[57]: - y_S[57] <= -113.4
 c_min_S[58]: - y_S[58] <= -113.4
 c_min_S[59]: - y_S[59] <= -113.4
 c_min_S[60]: - y_S[60] <= -113.4
 c_min_S[61]: - y_S[61] <= -113.4
 c_min_S[62]: - y_S[62] <= -113.4
 c_min_S[63]: - y_S[63] <= -113.4
 c_min_S[64]: - y_S[64] <= -113.4
 c_min_S[65]: - y_S[65] <= -113.4
 c_min_S[66]: - y_S[66] <= -113.4
 c_min_S[67]: - y_S[67] <= -113.4
 c_min_S[68]: - y_S[68] <= -113.4
 c_min_S[69]: - y_S[69] <= -113.4
 c_min_S[70]: - y_S[70] <= -113.4
 c_min_S[71]: - y_S[71] <= -113.4
 c_min_S[72]: - y_S[72] <= -113.4
 c_min_S[73]: - y_S[73] <= -113.4
 c_min_S[74]: - y_S[74] <= -113.4
 c_min_S[75]: - y_S[75] <= -113.4
 c_min_S[76]: - y_S[76] <= -113.4
 c_min_S[77]: - y_S[77] <= -113.4
 c_min_S[78]: - y_S[78] <= -113.4
 c_min_S[79]: - y_S[79] <= -113.4
 c_min_S[80]: - y_S[80] <= -113.4
 c_min_S[81]: - y_S[81] <= -113.4
 c_min_S[82]: - y_S[82] <= -113.4
 c_min_S[83]: - y_S[83] <= -113.4
 c_min_S[84]: - y_S[84] <= -113.4
 c_min_S[85]: - y_S[85] <= -113.4
 c_min_S[86]: - y_S[86] <= -113.4
 c_min_S[87]: - y_S[87] <= -113.4
 c_min_S[88]: - y_S[88] <= -113.4
 c_min_S[89]: - y_S[89] <= -113.4
 c_min_S[90]: - y_S[90] <= -113.4
 c_min_S[91]: - y_S[91] <= -113.4
 c_min_S[92]: - y_S[92] <= -113.4
 c_min_S[93]: - y_S[93] <= -113.4
 c_min_S[94]: - y_S[94] <= -113.4
 c_min_S[95]: - y_S[95] <= -113.4
 c_max_S[0]: y_S[0] <= 453.6
 c_max_S[1]: y_S[1] <= 453.6
 c_max_S[2]: y_S[2] <= 453.6
 c_max_S[3]: y_S[3] <= 453.6
 c_max_S[4]: y_S[4] <= 453.6
 c_max_S[5]: y_S[5] <= 453.6
 c_max_S[6]: y_S[6] <= 453.6
 c_max_S[7]: y_S[7] <= 453.6
 c_max_S[8]: y_S[8] <= 453.6
 c_max_S[9]: y_S[9] <= 453.6
 c_max_S[10]: y_S[10] <= 453.6
 c_max_S[11]: y_S[11] <= 453.6
 c_max_S[12]: y_S[12] <= 453.6
 c_max_S[13]: y_S[13] <= 453.6
 c_max_S[14]: y_S[14] <= 453.6
 c_max_S[15]: y_S[15] <= 453.6
 c_max_S[16]: y_S[16] <= 453.6
 c_max_S[17]: y_S[17] <= 453.6
 c_max_S[18]: y_S[18] <= 453.6
 c_max_S[19]: y_S[19] <= 453.6
 c_max_S[20]: y_S[20] <= 453.6
 c_max_S[21]: y_S[21] <= 453.6
 c_max_S[22]: y_S[22] <= 453.6
 c_max_S[23]: y_S[23] <= 453.6
 c_max_S[24]: y_S[24] <= 453.6
 c_max_S[25]: y_S[25] <= 453.6
 c_max_S[26]: y_S[26] <= 453.6
 c_max_S[27]: y_S[27] <= 453.6
 c_max_S[28]: y_S[28] <= 453.6
 c_max_S[29]: y_S[29] <= 453.6
 c_max_S[30]: y_S[30] <= 453.6
 c_max_S[31]: y_S[31] <= 453.6
 c_max_S[32]: y_S[32] <= 453.6
 c_max_S[33]: y_S[33] <= 453.6
 c_max_S[34]: y_S[34] <= 453.6
 c_max_S[35]: y_S[35] <= 453.6
 c_max_S[36]: y_S[36] <= 453.6
 c_max_S[37]: y_S[37] <= 453.6
 c_max_S[38]: y_S[38] <= 453.6
 c_max_S[39]: y_S[39] <= 453.6
 c_max_S[40]: y_S[40] <= 453.6
 c_max_S[41]: y_S[41] <= 453.6
 c_max_S[42]: y_S[42] <= 453.6
 c_max_S[43]: y_S[43] <= 453.6
 c_max_S[44]: y_S[44] <= 453.6
 c_max_S[45]: y_S[45] <= 453.6
 c_max_S[46]: y_S[46] <= 453.6
 c_max_S[47]: y_S[47] <= 453.6
 c_max_S[48]: y_S[48] <= 453.6
 c_max_S[49]: y_S[49] <= 453.6
 c_max_S[50]: y_S[50] <= 453.6
 c_max_S[51]: y_S[51] <= 453.6
 c_max_S[52]: y_S[52] <= 453.6
 c_max_S[53]: y_S[53] <= 453.6
 c_max_S[54]: y_S[54] <= 453.6
 c_max_S[55]: y_S[55] <= 453.6
 c_max_S[56]: y_S[56] <= 453.6
 c_max_S[57]: y_S[57] <= 453.6
 c_max_S[58]: y_S[58] <= 453.6
 c_max_S[59]: y_S[59] <= 453.6
 c_max_S[60]: y_S[60] <= 453.6
 c_max_S[61]: y_S[61] <= 453.6
 c_max_S[62]: y_S[62] <= 453.6
 c_max_S[63]: y_S[63] <= 453.6
 c_max_S[64]: y_S[64] <= 453.6
 c_max_S[65]: y_S[65] <= 453.6
 c_max_S[66]: y_S[66] <= 453.6
 c_max_S[67]: y_S[67] <= 453.6
 c_max_S[68]: y_S[68] <= 453.6
 c_max_S[69]: y_S[69] <= 453.6
 c_max_S[70]: y_S[70] <= 453.6
 c_max_S[71]: y_S[71] <= 453.6
 c_max_S[72]: y_S[72] <= 453.6
 c_max_S[73]: y_S[73] <= 453.6
 c_max_S[74]: y_S[74] <= 453.6
 c_max_S[75]: y_S[75] <= 453.6
 c_max_S[76]: y_S[76] <= 453.6
 c_max_S[77]: y_S[77] <= 453.6
 c_max_S[78]: y_S[78] <= 453.6
 c_max_S[79]: y_S[79] <= 453.6
 c_max_S[80]: y_S[80] <= 453.6
 c_max_S[81]: y_S[81] <= 453.6
 c_max_S[82]: y_S[82] <= 453.6
 c_max_S[83]: y_S[83] <= 453.6
 c_max_S[84]: y_S[84] <= 453.6
 c_max_S[85]: y_S[85] <= 453.6
 c_max_S[86]: y_S[86] <= 453.6
 c_max_S[87]: y_S[87] <= 453.6
 c_max_S[88]: y_S[88] <= 453.6
 c_max_S[89]: y_S[89] <= 453.6
 c_max_S[90]: y_S[90] <= 453.6
 c_max_S[91]: y_S[91] <= 453.6
 c_max_S[92]: y_S[92] <= 453.6
 c_max_S[93]: y_S[93] <= 453.6
 c_max_S[94]: y_S[94] <= 453.6
 c_max_S[95]: y_S[95] <= 453.6
 c_ESS_first_period: - 0.2325 y_chg[0] + 0.2688172043010753 y_dis[0]
   + y_S[0] = 187.5
 c_ESS_re-dispatch[1]: - 0.2325 x_chg[1] + 0.2688172043010753 x_dis[1]
   - 0.2325 y_chg[1] + 0.2688172043010753 y_dis[1] - y_S[0] + y_S[1] = 0
 c_ESS_re-dispatch[2]: - 0.2325 x_chg[2] + 0.2688172043010753 x_dis[2]
   - 0.2325 y_chg[2] + 0.2688172043010753 y_dis[2] - y_S[1] + y_S[2] = 0
 c_ESS_re-dispatch[3]: - 0.2325 x_chg[3] + 0.2688172043010753 x_dis[3]
   - 0.2325 y_chg[3] + 0.2688172043010753 y_dis[3] - y_S[2] + y_S[3] = 0
 c_ESS_re-dispatch[4]: - 0.2325 x_chg[4] + 0.2688172043010753 x_dis[4]
   - 0.2325 y_chg[4] + 0.2688172043010753 y_dis[4] - y_S[3] + y_S[4] = 0
 c_ESS_re-dispatch[5]: - 0.2325 x_chg[5] + 0.2688172043010753 x_dis[5]
   - 0.2325 y_chg[5] + 0.2688172043010753 y_dis[5] - y_S[4] + y_S[5] = 0
 c_ESS_re-dispatch[6]: - 0.2325 x_chg[6] + 0.2688172043010753 x_dis[6]
   - 0.2325 y_chg[6] + 0.2688172043010753 y_dis[6] - y_S[5] + y_S[6] = 0
 c_ESS_re-dispatch[7]: - 0.2325 x_chg[7] + 0.2688172043010753 x_dis[7]
   - 0.2325 y_chg[7] + 0.2688172043010753 y_dis[7] - y_S[6] + y_S[7] = 0
 c_ESS_re-dispatch[8]: - 0.2325 x_chg[8] + 0.2688172043010753 x_dis[8]
   - 0.2325 y_chg[8] + 0.2688172043010753 y_dis[8] - y_S[7] + y_S[8] = 0
 c_ESS_re-dispatch[9]: - 0.2325 x_chg[9] + 0.2688172043010753 x_dis[9]
   - 0.2325 y_chg[9] + 0.2688172043010753 y_dis[9] - y_S[8] + y_S[9] = 0
 c_ESS_re-dispatch[10]: - 0.2325 x_chg[10] + 0.2688172043010753 x_dis[10]
   - 0.2325 y_chg[10] + 0.2688172043010753 y_dis[10] - y_S[9] + y_S[10]
   = 0
 c_ESS_re-dispatch[11]: - 0.2325 x_chg[11] + 0.2688172043010753 x_dis[11]
   - 0.2325 y_chg[11] + 0.2688172043010753 y_dis[11] - y_S[10] + y_S[11]
   = 0
 c_ESS_re-dispatch[12]: - 0.2325 x_chg[12] + 0.2688172043010753 x_dis[12]
   - 0.2325 y_chg[12] + 0.2688172043010753 y_dis[12] - y_S[11] + y_S[12]
   = 0
 c_ESS_re-dispatch[13]: - 0.2325 x_chg[13] + 0.2688172043010753 x_dis[13]
   - 0.2325 y_chg[13] + 0.2688172043010753 y_dis[13] - y_S[12] + y_S[13]
   = 0
 c_ESS_re-dispatch[14]: - 0.2325 x_chg[14] + 0.2688172043010753 x_dis[14]
   - 0.2325 y_chg[14] + 0.2688172043010753 y_dis[14] - y_S[13] + y_S[14]
   = 0
 c_ESS_re-dispatch[15]: - 0.2325 x_chg[15] + 0.2688172043010753 x_dis[15]
   - 0.2325 y_chg[15] + 0.2688172043010753 y_dis[15] - y_S[14] + y_S[15]
   = 0
 c_ESS_re-dispatch[16]: - 0.2325 x_chg[16] + 0.2688172043010753 x_dis[16]
   - 0.2325 y_chg[16] + 0.2688172043010753 y_dis[16] - y_S[15] + y_S[16]
   = 0
 c_ESS_re-dispatch[17]: - 0.2325 x_chg[17] + 0.2688172043010753 x_dis[17]
   - 0.2325 y_chg[17] + 0.2688172043010753 y_dis[17] - y_S[16] + y_S[17]
   = 0
 c_ESS_re-dispatch[18]: - 0.2325 x_chg[18] + 0.2688172043010753 x_dis[18]
   - 0.2325 y_chg[18] + 0.2688172043010753 y_dis[18] - y_S[17] + y_S[18]
   = 0
 c_ESS_re-dispatch[19]: - 0.2325 x_chg[19] + 0.2688172043010753 x_dis[19]
   - 0.2325 y_chg[19] + 0.2688172043010753 y_dis[19] - y_S[18] + y_S[19]
   = 0
 c_ESS_re-dispatch[20]: - 0.2325 x_chg[20] + 0.2688172043010753 x_dis[20]
   - 0.2325 y_chg[20] + 0.2688172043010753 y_dis[20] - y_S[19] + y_S[20]
   = 0
 c_ESS_re-dispatch[21]: - 0.2325 x_chg[21] + 0.2688172043010753 x_dis[21]
   - 0.2325 y_chg[21] + 0.2688172043010753 y_dis[21] - y_S[20] + y_S[21]
   = 0
 c_ESS_re-dispatch[22]: - 0.2325 x_chg[22] + 0.2688172043010753 x_dis[22]
   - 0.2325 y_chg[22] + 0.2688172043010753 y_dis[22] - y_S[21] + y_S[22]
   = 0
 c_ESS_re-dispatch[23]: - 0.2325 x_chg[23] + 0.2688172043010753 x_dis[23]
   - 0.2325 y_chg[23] + 0.2688172043010753 y_dis[23] - y_S[22] + y_S[23]
   = 0
 c_ESS_re-dispatch[24]: - 0.2325 x_chg[24] + 0.2688172043010753 x_dis[24]
   - 0.2325 y_chg[24] + 0.2688172043010753 y_dis[24] - y_S[23] + y_S[24]
   = 0
 c_ESS_re-dispatch[25]: - 0.2325 x_chg[25] + 0.2688172043010753 x_dis[25]
   - 0.2325 y_chg[25] + 0.2688172043010753 y_dis[25] - y_S[24] + y_S[25]
   = 0
 c_ESS_re-dispatch[26]: - 0.2325 x_chg[26] + 0.2688172043010753 x_dis[26]
   - 0.2325 y_chg[26] + 0.2688172043010753 y_dis[26] - y_S[25] + y_S[26]
   = 0
 c_ESS_re-dispatch[27]: - 0.2325 x_chg[27] + 0.2688172043010753 x_dis[27]
   - 0.2325 y_chg[27] + 0.2688172043010753 y_dis[27] - y_S[26] + y_S[27]
   = 0
 c_ESS_re-dispatch[28]: - 0.2325 x_chg[28] + 0.2688172043010753 x_dis[28]
   - 0.2325 y_chg[28] + 0.2688172043010753 y_dis[28] - y_S[27] + y_S[28]
   = 0
 c_ESS_re-dispatch[29]: - 0.2325 x_chg[29] + 0.2688172043010753 x_dis[29]
   - 0.2325 y_chg[29] + 0.2688172043010753 y_dis[29] - y_S[28] + y_S[29]
   = 0
 c_ESS_re-dispatch[30]: - 0.2325 x_chg[30] + 0.2688172043010753 x_dis[30]
   - 0.2325 y_chg[30] + 0.2688172043010753 y_dis[30] - y_S[29] + y_S[30]
   = 0
 c_ESS_re-dispatch[31]: - 0.2325 x_chg[31] + 0.2688172043010753 x_dis[31]
   - 0.2325 y_chg[31] + 0.2688172043010753 y_dis[31] - y_S[30] + y_S[31]
   = 0
 c_ESS_re-dispatch[32]: - 0.2325 x_chg[32] + 0.2688172043010753 x_dis[32]
   - 0.2325 y_chg[32] + 0.2688172043010753 y_dis[32] - y_S[31] + y_S[32]
   = 0
 c_ESS_re-dispatch[33]: - 0.2325 x_chg[33] + 0.2688172043010753 x_dis[33]
   - 0.2325 y_chg[33] + 0.2688172043010753 y_dis[33] - y_S[32] + y_S[33]
   = 0
 c_ESS_re-dispatch[34]: - 0.2325 x_chg[34] + 0.2688172043010753 x_dis[34]
   - 0.2325 y_chg[34] + 0.2688172043010753 y_dis[34] - y_S[33] + y_S[34]
   = 0
 c_ESS_re-dispatch[35]: - 0.2325 x_chg[35] + 0.2688172043010753 x_dis[35]
   - 0.2325 y_chg[35] + 0.2688172043010753 y_dis[35] - y_S[34] + y_S[35]
   = 0
 c_ESS_re-dispatch[36]: - 0.2325 x_chg[36] + 0.2688172043010753 x_dis[36]
   - 0.2325 y_chg[36] + 0.2688172043010753 y_dis[36] - y_S[35] + y_S[36]
   = 0
 c_ESS_re-dispatch[37]: - 0.2325 x_chg[37] + 0.2688172043010753 x_dis[37]
   - 0.2325 y_chg[37] + 0.2688172043010753 y_dis[37] - y_S[36] + y_S[37]
   = 0
 c_ESS_re-dispatch[38]: - 0.2325 x_chg[38] + 0.2688172043010753 x_dis[38]
   - 0.2325 y_chg[38] + 0.2688172043010753 y_dis[38] - y_S[37] + y_S[38]
   = 0
 c_ESS_re-dispatch[39]: - 0.2325 x_chg[39] + 0.2688172043010753 x_dis[39]
   - 0.2325 y_chg[39] + 0.2688172043010753 y_dis[39] - y_S[38] + y_S[39]
   = 0
 c_ESS_re-dispatch[40]: - 0.2325 x_chg[40] + 0.2688172043010753 x_dis[40]
   - 0.2325 y_chg[40] + 0.2688172043010753 y_dis[40] - y_S[39] + y_S[40]
   = 0
 c_ESS_re-dispatch[41]: - 0.2325 x_chg[41] + 0.2688172043010753 x_dis[41]
   - 0.2325 y_chg[41] + 0.2688172043010753 y_dis[41] - y_S[40] + y_S[41]
   = 0
 c_ESS_re-dispatch[42]: - 0.2325 x_chg[42] + 0.2688172043010753 x_dis[42]
   - 0.2325 y_chg[42] + 0.2688172043010753 y_dis[42] - y_S[41] + y_S[42]
   = 0
 c_ESS_re-dispatch[43]: - 0.2325 x_chg[43] + 0.2688172043010753 x_dis[43]
   - 0.2325 y_chg[43] + 0.2688172043010753 y_dis[43] - y_S[42] + y_S[43]
   = 0
 c_ESS_re-dispatch[44]: - 0.2325 x_chg[44] + 0.2688172043010753 x_dis[44]
   - 0.2325 y_chg[44] + 0.2688172043010753 y_dis[44] - y_S[43] + y_S[44]
   = 0
 c_ESS_re-dispatch[45]: - 0.2325 x_chg[45] + 0.2688172043010753 x_dis[45]
   - 0.2325 y_chg[45] + 0.2688172043010753 y_dis[45] - y_S[44] + y_S[45]
   = 0
 c_ESS_re-dispatch[46]: - 0.2325 x_chg[46] + 0.2688172043010753 x_dis[46]
   - 0.2325 y_chg[46] + 0.2688172043010753 y_dis[46] - y_S[45] + y_S[46]
   = 0
 c_ESS_re-dispatch[47]: - 0.2325 x_chg[47] + 0.2688172043010753 x_dis[47]
   - 0.2325 y_chg[47] + 0.2688172043010753 y_dis[47] - y_S[46] + y_S[47]
   = 0
 c_ESS_re-dispatch[48]: - 0.2325 x_chg[48] + 0.2688172043010753 x_dis[48]
   - 0.2325 y_chg[48] + 0.2688172043010753 y_dis[48] - y_S[47] + y_S[48]
   = 0
 c_ESS_re-dispatch[49]: - 0.2325 x_chg[49] + 0.2688172043010753 x_dis[49]
   - 0.2325 y_chg[49] + 0.2688172043010753 y_dis[49] - y_S[48] + y_S[49]
   = 0
 c_ESS_re-dispatch[50]: - 0.2325 x_chg[50] + 0.2688172043010753 x_dis[50]
   - 0.2325 y_chg[50] + 0.2688172043010753 y_dis[50] - y_S[49] + y_S[50]
   = 0
 c_ESS_re-dispatch[51]: - 0.2325 x_chg[51] + 0.2688172043010753 x_dis[51]
   - 0.2325 y_chg[51] + 0.2688172043010753 y_dis[51] - y_S[50] + y_S[51]
   = 0
 c_ESS_re-dispatch[52]: - 0.2325 x_chg[52] + 0.2688172043010753 x_dis[52]
   - 0.2325 y_chg[52] + 0.2688172043010753 y_dis[52] - y_S[51] + y_S[52]
   = 0
 c_ESS_re-dispatch[53]: - 0.2325 x_chg[53] + 0.2688172043010753 x_dis[53]
   - 0.2325 y_chg[53] + 0.2688172043010753 y_dis[53] - y_S[52] + y_S[53]
   = 0
 c_ESS_re-dispatch[54]: - 0.2325 x_chg[54] + 0.2688172043010753 x_dis[54]
   - 0.2325 y_chg[54] + 0.2688172043010753 y_dis[54] - y_S[53] + y_S[54]
   = 0
 c_ESS_re-dispatch[55]: - 0.2325 x_chg[55] + 0.2688172043010753 x_dis[55]
   - 0.2325 y_chg[55] + 0.2688172043010753 y_dis[55] - y_S[54] + y_S[55]
   = 0
 c_ESS_re-dispatch[56]: - 0.2325 x_chg[56] + 0.2688172043010753 x_dis[56]
   - 0.2325 y_chg[56] + 0.2688172043010753 y_dis[56] - y_S[55] + y_S[56]
   = 0
 c_ESS_re-dispatch[57]: - 0.2325 x_chg[57] + 0.2688172043010753 x_dis[57]
   - 0.2325 y_chg[57] + 0.2688172043010753 y_dis[57] - y_S[56] + y_S[57]
   = 0
 c_ESS_re-dispatch[58]: - 0.2325 x_chg[58] + 0.2688172043010753 x_dis[58]
   - 0.2325 y_chg[58] + 0.2688172043010753 y_dis[58] - y_S[57] + y_S[58]
   = 0
 c_ESS_re-dispatch[59]: - 0.2325 x_chg[59] + 0.2688172043010753 x_dis[59]
   - 0.2325 y_chg[59] + 0.2688172043010753 y_dis[59] - y_S[58] + y_S[59]
   = 0
 c_ESS_re-dispatch[60]: - 0.2325 x_chg[60] + 0.2688172043010753 x_dis[60]
   - 0.2325 y_chg[60] + 0.2688172043010753 y_dis[60] - y_S[59] + y_S[60]
   = 0
 c_ESS_re-dispatch[61]: - 0.2325 x_chg[61] + 0.2688172043010753 x_dis[61]
   - 0.2325 y_chg[61] + 0.2688172043010753 y_dis[61] - y_S[60] + y_S[61]
   = 0
 c_ESS_re-dispatch[62]: - 0.2325 x_chg[62] + 0.2688172043010753 x_dis[62]
   - 0.2325 y_chg[62] + 0.2688172043010753 y_dis[62] - y_S[61] + y_S[62]
   = 0
 c_ESS_re-dispatch[63]: - 0.2325 x_chg[63] + 0.2688172043010753 x_dis[63]
   - 0.2325 y_chg[63] + 0.2688172043010753 y_dis[63] - y_S[62] + y_S[63]
   = 0
 c_ESS_re-dispatch[64]: - 0.2325 x_chg[64] + 0.2688172043010753 x_dis[64]
   - 0.2325 y_chg[64] + 0.2688172043010753 y_dis[64] - y_S[63] + y_S[64]
   = 0
 c_ESS_re-dispatch[65]: - 0.2325 x_chg[65] + 0.2688172043010753 x_dis[65]
   - 0.2325 y_chg[65] + 0.2688172043010753 y_dis[65] - y_S[64] + y_S[65]
   = 0
 c_ESS_re-dispatch[66]: - 0.2325 x_chg[66] + 0.2688172043010753 x_dis[66]
   - 0.2325 y_chg[66] + 0.2688172043010753 y_dis[66] - y_S[65] + y_S[66]
   = 0
 c_ESS_re-dispatch[67]: - 0.2325 x_chg[67] + 0.2688172043010753 x_dis[67]
   - 0.2325 y_chg[67] + 0.2688172043010753 y_dis[67] - y_S[66] + y_S[67]
   = 0
 c_ESS_re-dispatch[68]: - 0.2325 x_chg[68] + 0.2688172043010753 x_dis[68]
   - 0.2325 y_chg[68] + 0.2688172043010753 y_dis[68] - y_S[67] + y_S[68]
   = 0
 c_ESS_re-dispatch[69]: - 0.2325 x_chg[69] + 0.2688172043010753 x_dis[69]
   - 0.2325 y_chg[69] + 0.2688172043010753 y_dis[69] - y_S[68] + y_S[69]
   = 0
 c_ESS_re-dispatch[70]: - 0.2325 x_chg[70] + 0.2688172043010753 x_dis[70]
   - 0.2325 y_chg[70] + 0.2688172043010753 y_dis[70] - y_S[69] + y_S[70]
   = 0
 c_ESS_re-dispatch[71]: - 0.2325 x_chg[71] + 0.2688172043010753 x_dis[71]
   - 0.2325 y_chg[71] + 0.2688172043010753 y_dis[71] - y_S[70] + y_S[71]
   = 0
 c_ESS_re-dispatch[72]: - 0.2325 x_chg[72] + 0.2688172043010753 x_dis[72]
   - 0.2325 y_chg[72] + 0.2688172043010753 y_dis[72] - y_S[71] + y_S[72]
   = 0
 c_ESS_re-dispatch[73]: - 0.2325 x_chg[73] + 0.2688172043010753 x_dis[73]
   - 0.2325 y_chg[73] + 0.2688172043010753 y_dis[73] - y_S[72] + y_S[73]
   = 0
 c_ESS_re-dispatch[74]: - 0.2325 x_chg[74] + 0.2688172043010753 x_dis[74]
   - 0.2325 y_chg[74] + 0.2688172043010753 y_dis[74] - y_S[73] + y_S[74]
   = 0
 c_ESS_re-dispatch[75]: - 0.2325 x_chg[75] + 0.2688172043010753 x_dis[75]
   - 0.2325 y_chg[75] + 0.2688172043010753 y_dis[75] - y_S[74] + y_S[75]
   = 0
 c_ESS_re-dispatch[76]: - 0.2325 x_chg[76] + 0.2688172043010753 x_dis[76]
   - 0.2325 y_chg[76] + 0.2688172043010753 y_dis[76] - y_S[75] + y_S[76]
   = 0
 c_ESS_re-dispatch[77]: - 0.2325 x_chg[77] + 0.2688172043010753 x_dis[77]
   - 0.2325 y_chg[77] + 0.2688172043010753 y_dis[77] - y_S[76] + y_S[77]
   = 0
 c_ESS_re-dispatch[78]: - 0.2325 x_chg[78] + 0.2688172043010753 x_dis[78]
   - 0.2325 y_chg[78] + 0.2688172043010753 y_dis[78] - y_S[77] + y_S[78]
   = 0
 c_ESS_re-dispatch[79]: - 0.2325 x_chg[79] + 0.2688172043010753 x_dis[79]
   - 0.2325 y_chg[79] + 0.2688172043010753 y_dis[79] - y_S[78] + y_S[79]
   = 0
 c_ESS_re-dispatch[80]: - 0.2325 x_chg[80] + 0.2688172043010753 x_dis[80]
   - 0.2325 y_chg[80] + 0.2688172043010753 y_dis[80] - y_S[79] + y_S[80]
   = 0
 c_ESS_re-dispatch[81]: - 0.2325 x_chg[81] + 0.2688172043010753 x_dis[81]
   - 0.2325 y_chg[81] + 0.2688172043010753 y_dis[81] - y_S[80] + y_S[81]
   = 0
 c_ESS_re-dispatch[82]: - 0.2325 x_chg[82] + 0.2688172043010753 x_dis[82]
   - 0.2325 y_chg[82] + 0.2688172043010753 y_dis[82] - y_S[81] + y_S[82]
   = 0
 c_ESS_re-dispatch[83]: - 0.2325 x_chg[83] + 0.2688172043010753 x_dis[83]
   - 0.2325 y_chg[83] + 0.2688172043010753 y_dis[83] - y_S[82] + y_S[83]
   = 0
 c_ESS_re-dispatch[84]: - 0.2325 x_chg[84] + 0.2688172043010753 x_dis[84]
   - 0.2325 y_chg[84] + 0.2688172043010753 y_dis[84] - y_S[83] + y_S[84]
   = 0
 c_ESS_re-dispatch[85]: - 0.2325 x_chg[85] + 0.2688172043010753 x_dis[85]
   - 0.2325 y_chg[85] + 0.2688172043010753 y_dis[85] - y_S[84] + y_S[85]
   = 0
 c_ESS_re-dispatch[86]: - 0.2325 x_chg[86] + 0.2688172043010753 x_dis[86]
   - 0.2325 y_chg[86] + 0.2688172043010753 y_dis[86] - y_S[85] + y_S[86]
   = 0
 c_ESS_re-dispatch[87]: - 0.2325 x_chg[87] + 0.2688172043010753 x_dis[87]
   - 0.2325 y_chg[87] + 0.2688172043010753 y_dis[87] - y_S[86] + y_S[87]
   = 0
 c_ESS_re-dispatch[88]: - 0.2325 x_chg[88] + 0.2688172043010753 x_dis[88]
   - 0.2325 y_chg[88] + 0.2688172043010753 y_dis[88] - y_S[87] + y_S[88]
   = 0
 c_ESS_re-dispatch[89]: - 0.2325 x_chg[89] + 0.2688172043010753 x_dis[89]
   - 0.2325 y_chg[89] + 0.2688172043010753 y_dis[89] - y_S[88] + y_S[89]
   = 0
 c_ESS_re-dispatch[90]: - 0.2325 x_chg[90] + 0.2688172043010753 x_dis[90]
   - 0.2325 y_chg[90] + 0.2688172043010753 y_dis[90] - y_S[89] + y_S[90]
   = 0
 c_ESS_re-dispatch[91]: - 0.2325 x_chg[91] + 0.2688172043010753 x_dis[91]
   - 0.2325 y_chg[91] + 0.2688172043010753 y_dis[91] - y_S[90] + y_S[91]
   = 0
 c_ESS_re-dispatch[92]: - 0.2325 x_chg[92] + 0.2688172043010753 x_dis[92]
   - 0.2325 y_chg[92] + 0.2688172043010753 y_dis[92] - y_S[91] + y_S[92]
   = 0
 c_ESS_re-dispatch[93]: - 0.2325 x_chg[93] + 0.2688172043010753 x_dis[93]
   - 0.2325 y_chg[93] + 0.2688172043010753 y_dis[93] - y_S[92] + y_S[93]
   = 0
 c_ESS_re-dispatch[94]: - 0.2325 x_chg[94] + 0.2688172043010753 x_dis[94]
   - 0.2325 y_chg[94] + 0.2688172043010753 y_dis[94] - y_S[93] + y_S[94]
   = 0
 c_ESS_re-dispatch[95]: - 0.2325 x_chg[95] + 0.2688172043010753 x_dis[95]
   - 0.2325 y_chg[95] + 0.2688172043010753 y_dis[95] - y_S[94] + y_S[95]
   = 0
 c_ESS_last_period: y_S[95] = 187.5
 c_PV_re-dispatch[0]: y_PV[0] = 0
 c_PV_re-dispatch[1]: y_PV[1] = 0
 c_PV_re-dispatch[2]: y_PV[2] = 0
 c_PV_re-dispatch[3]: y_PV[3] = 0
 c_PV_re-dispatch[4]: y_PV[4] = 0
 c_PV_re-dispatch[5]: y_PV[5] = 0
 c_PV_re-dispatch[6]: y_PV[6] = 0
 c_PV_re-dispatch[7]: y_PV[7] = 0
 c_PV_re-dispatch[8]: y_PV[8] = 0
 c_PV_re-dispatch[9]: y_PV[9] = 0
 c_PV_re-dispatch[10]: y_PV[10] = 0
 c_PV_re-dispatch[11]: y_PV[11] = 0
 c_PV_re-dispatch[12]: y_PV[12] = 0
 c_PV_re-dispatch[13]: y_PV[13] = 0
 c_PV_re-dispatch[14]: y_PV[14] = 0
 c_PV_re-dispatch[15]: y_PV[15] = 0
 c_PV_re-dispatch[16]: y_PV[16] = 0
 c_PV_re-dispatch[17]: y_PV[17] = 0
 c_PV_re-dispatch[18]: y_PV[18] = 0
 c_PV_re-dispatch[19]: y_PV[19] = 0
 c_PV_re-dispatch[20]: y_PV[20] = 0
 c_PV_re-dispatch[21]: y_PV[21] = 0
 c_PV_re-dispatch[22]: y_PV[22] = 0
 c_PV_re-dispatch[23]: y_PV[23] = 0
 c_PV_re-dispatch[24]: y_PV[24] = 0
 c_PV_re-dispatch[25]: y_PV[25] = 0
 c_PV_re-dispatch[26]: y_PV[26] = 0
 c_PV_re-dispatch[27]: y_PV[27] = 0
 c_PV_re-dispatch[28]: y_PV[28] = 0
 c_PV_re-dispatch[29]: y_PV[29] = 0.6817549108937482
 c_PV_re-dispatch[30]: y_PV[30] = 3.927929373725216
 c_PV_re-dispatch[31]: y_PV[31] = 12.01863468178648
 c_PV_re-dispatch[32]: y_PV[32] = 30.51313193721121
 c_PV_re-dispatch[33]: y_PV[33] = 35.08499965125498
 c_PV_re-dispatch[34]: y_PV[34] = 55.17332513737617
 c_PV_re-dispatch[35]: y_PV[35] = 140.1686278350507
 c_PV_re-dispatch[36]: y_PV[36] = 178.2836440688019
 c_PV_re-dispatch[37]: y_PV[37] = 207.9424740079004
 c_PV_re-dispatch[38]: y_PV[38] = 232.8479934620799
 c_PV_re-dispatch[39]: y_PV[39] = 263.3768905490792
 c_PV_re-dispatch[40]: y_PV[40] = 287.0985454425697
 c_PV_re-dispatch[41]: y_PV[41] = 309.8822349462725
 c_PV_re-dispatch[42]: y_PV[42] = 333.2777732532034
 c_PV_re-dispatch[43]: y_PV[43] = 352.389781719958
 c_PV_re-dispatch[44]: y_PV[44] = 375.7159161102198
 c_PV_re-dispatch[45]: y_PV[45] = 394.9875487973123
 c_PV_re-dispatch[46]: y_PV[46] = 409.7917531813302
 c_PV_re-dispatch[47]: y_PV[47] = 428.101815569561
 c_PV_re-dispatch[48]: y_PV[48] = 440.6464964382849
 c_PV_re-dispatch[49]: y_PV[49] = 445.3786620241449
 c_PV_re-dispatch[50]: y_PV[50] = 455.556472272189
 c_PV_re-dispatch[51]: y_PV[51] = 460.5117236642729
 c_PV_re-dispatch[52]: y_PV[52] = 464.00084206478
 c_PV_re-dispatch[53]: y_PV[53] = 469.2390762828775
 c_PV_re-dispatch[54]: y_PV[54] = 476.6880525993341
 c_PV_re-dispatch[55]: y_PV[55] = 473.7111903963382
 c_PV_re-dispatch[56]: y_PV[56] = 455.8462723276253
 c_PV_re-dispatch[57]: y_PV[57] = 436.6431681623882
 c_PV_re-dispatch[58]: y_PV[58] = 414.4684218590571
 c_PV_re-dispatch[59]: y_PV[59] = 393.7351348159873
 c_PV_re-dispatch[60]: y_PV[60] = 290.1728620584943
 c_PV_re-dispatch[61]: y_PV[61] = 350.6164410277992
 c_PV_re-dispatch[62]: y_PV[62] = 322.9022729394957
 c_PV_re-dispatch[63]: y_PV[63] = 293.0620625079054
 c_PV_re-dispatch[64]: y_PV[64] = 262.7833201991659
 c_PV_re-dispatch[65]: y_PV[65] = 231.385172051662
 c_PV_re-dispatch[66]: y_PV[66] = 199.9895743469131
 c_PV_re-dispatch[67]: y_PV[67] = 165.7915647075124
 c_PV_re-dispatch[68]: y_PV[68] = 132.5896733187977
 c_PV_re-dispatch[69]: y_PV[69] = 98.15859347428747
 c_PV_re-dispatch[70]: y_PV[70] = 40.61665929262848
 c_PV_re-dispatch[71]: y_PV[71] = 30.04096046923865
 c_PV_re-dispatch[72]: y_PV[72] = 16.20481073549318
 c_PV_re-dispatch[73]: y_PV[73] = 2.626139118086846
 c_PV_re-dispatch[74]: y_PV[74] = 0
 c_PV_re-dispatch[75]: y_PV[75] = 0
 c_PV_re-dispatch[76]: y_PV[76] = 0
 c_PV_re-dispatch[77]: y_PV[77] = 0
 c_PV_re-dispatch[78]: y_PV[78] = 0
 c_PV_re-dispatch[79]: y_PV[79] = 0
 c_PV_re-dispatch[80]: y_PV[80] = 0
 c_PV_re-dispatch[81]: y_PV[81] = 0
 c_PV_re-dispatch[82]: y_PV[82] = 0
 c_PV_re-dispatch[83]: y_PV[83] = 0
 c_PV_re-dispatch[84]: y_PV[84] = 0
 c_PV_re-dispatch[85]: y_PV[85] = 0
 c_PV_re-dispatch[86]: y_PV[86] = 0
 c_PV_re-dispatch[87]: y_PV[87] = 0
 c_PV_re-dispatch[88]: y_PV[88] = 0
 c_PV_re-dispatch[89]: y_PV[89] = 0
 c_PV_re-dispatch[90]: y_PV[90] = 0
 c_PV_re-dispatch[91]: y_PV[91] = 0
 c_PV_re-dispatch[92]: y_PV[92] = 0
 c_PV_re-dispatch[93]: y_PV[93] = 0
 c_PV_re-dispatch[94]: y_PV[94] = 0
 c_PV_re-dispatch[95]: y_PV[95] = 0
 c_load_re-dispatch[0]: y_load[0] = 308.1600036621094
 c_load_re-dispatch[1]: y_load[1] = 306.4299926757813
 c_load_re-dispatch[2]: y_load[2] = 309.0700073242188
 c_load_re-dispatch[3]: y_load[3] = 307.8200073242188
 c_load_re-dispatch[4]: y_load[4] = 308.1600036621094
 c_load_re-dispatch[5]: y_load[5] = 310.4200134277344
 c_load_re-dispatch[6]: y_load[6] = 306.3800048828125
 c_load_re-dispatch[7]: y_load[7] = 309.9400024414063
 c_load_re-dispatch[8]: y_load[8] = 308.3999938964844
 c_load_re-dispatch[9]: y_load[9] = 309.5
 c_load_re-dispatch[10]: y_load[10] = 310.6600036621094
 c_load_re-dispatch[11]: y_load[11] = 309.8399963378906
 c_load_re-dispatch[12]: y_load[12] = 309.0199890136719
 c_load_re-dispatch[13]: y_load[13] = 311.6600036621094
 c_load_re-dispatch[14]: y_load[14] = 308.1099853515625
 c_load_re-dispatch[15]: y_load[15] = 311.9500122070313
 c_load_re-dispatch[16]: y_load[16] = 307.010009765625
 c_load_re-dispatch[17]: y_load[17] = 308.9800109863281
 c_load_re-dispatch[18]: y_load[18] = 308.1600036621094
 c_load_re-dispatch[19]: y_load[19] = 307.6300048828125
 c_load_re-dispatch[20]: y_load[20] = 308.7799987792969
 c_load_re-dispatch[21]: y_load[21] = 307.4400024414063
 c_load_re-dispatch[22]: y_load[22] = 308.0199890136719
 c_load_re-dispatch[23]: y_load[23] = 309.0700073242188
 c_load_re-dispatch[24]: y_load[24] = 306.5299987792969
 c_load_re-dispatch[25]: y_load[25] = 308.9299926757813
 c_load_re-dispatch[26]: y_load[26] = 305.5700073242188
 c_load_re-dispatch[27]: y_load[27] = 305.7099914550781
 c_load_re-dispatch[28]: y_load[28] = 308.3999938964844
 c_load_re-dispatch[29]: y_load[29] = 304.9400024414063
 c_load_re-dispatch[30]: y_load[30] = 306.3399963378906
 c_load_re-dispatch[31]: y_load[31] = 303.8399963378906
 c_load_re-dispatch[32]: y_load[32] = 302.3999938964844
 c_load_re-dispatch[33]: y_load[33] = 305.6600036621094
 c_load_re-dispatch[34]: y_load[34] = 303.8900146484375
 c_load_re-dispatch[35]: y_load[35] = 305.1799926757813
 c_load_re-dispatch[36]: y_load[36] = 306.5299987792969
 c_load_re-dispatch[37]: y_load[37] = 308.0599975585938
 c_load_re-dispatch[38]: y_load[38] = 307.3900146484375
 c_load_re-dispatch[39]: y_load[39] = 306.4800109863281
 c_load_re-dispatch[40]: y_load[40] = 307.25
 c_load_re-dispatch[41]: y_load[41] = 307.9200134277344
 c_load_re-dispatch[42]: y_load[42] = 305.1799926757813
 c_load_re-dispatch[43]: y_load[43] = 308.0599975585938
 c_load_re-dispatch[44]: y_load[44] = 307.3900146484375
 c_load_re-dispatch[45]: y_load[45] = 306.1400146484375
 c_load_re-dispatch[46]: y_load[46] = 307.7300109863281
 c_load_re-dispatch[47]: y_load[47] = 305.7099914550781
 c_load_re-dispatch[48]: y_load[48] = 306.5299987792969
 c_load_re-dispatch[49]: y_load[49] = 306.9599914550781
 c_load_re-dispatch[50]: y_load[50] = 305.3800048828125
 c_load_re-dispatch[51]: y_load[51] = 307.7799987792969
 c_load_re-dispatch[52]: y_load[52] = 306.1900024414063
 c_load_re-dispatch[53]: y_load[53] = 306.8200073242188
 c_load_re-dispatch[54]: y_load[54] = 307.010009765625
 c_load_re-dispatch[55]: y_load[55] = 305.9500122070313
 c_load_re-dispatch[56]: y_load[56] = 306.4299926757813
 c_load_re-dispatch[57]: y_load[57] = 308.739990234375
 c_load_re-dispatch[58]: y_load[58] = 306.7699890136719
 c_load_re-dispatch[59]: y_load[59] = 309.3099975585938
 c_load_re-dispatch[60]: y_load[60] = 305.4700012207031
 c_load_re-dispatch[61]: y_load[61] = 307.6300048828125
 c_load_re-dispatch[62]: y_load[62] = 305.5700073242188
 c_load_re-dispatch[63]: y_load[63] = 306.5299987792969
 c_load_re-dispatch[64]: y_load[64] = 306.2900085449219
 c_load_re-dispatch[65]: y_load[65] = 304.2200012207031
 c_load_re-dispatch[66]: y_load[66] = 303.0199890136719
 c_load_re-dispatch[67]: y_load[67] = 304.3699951171875
 c_load_re-dispatch[68]: y_load[68] = 303.3099975585938
 c_load_re-dispatch[69]: y_load[69] = 305.4700012207031
 c_load_re-dispatch[70]: y_load[70] = 304.989990234375
 c_load_re-dispatch[71]: y_load[71] = 304.7999877929688
 c_load_re-dispatch[72]: y_load[72] = 306.1400146484375
 c_load_re-dispatch[73]: y_load[73] = 307.7300109863281
 c_load_re-dispatch[74]: y_load[74] = 304.8999938964844
 c_load_re-dispatch[75]: y_load[75] = 304.7999877929688
 c_load_re-dispatch[76]: y_load[76] = 303.7000122070313
 c_load_re-dispatch[77]: y_load[77] = 304.0299987792969
 c_load_re-dispatch[78]: y_load[78] = 303.2200012207031
 c_load_re-dispatch[79]: y_load[79] = 303.6000061035156
 c_load_re-dispatch[80]: y_load[80] = 302.9800109863281
 c_load_re-dispatch[81]: y_load[81] = 302.5400085449219
 c_load_re-dispatch[82]: y_load[82] = 303.6000061035156
 c_load_re-dispatch[83]: y_load[83] = 302.260009765625
 c_load_re-dispatch[84]: y_load[84] = 303.2200012207031
 c_load_re-dispatch[85]: y_load[85] = 302.7799987792969
 c_load_re-dispatch[86]: y_load[86] = 302.6400146484375
 c_load_re-dispatch[87]: y_load[87] = 303.5499877929688
 c_load_re-dispatch[88]: y_load[88] = 302.8299865722656
 c_load_re-dispatch[89]: y_load[89] = 303.7000122070313
 c_load_re-dispatch[90]: y_load[90] = 304.0299987792969
 c_load_re-dispatch[91]: y_load[91] = 303.5
 c_load_re-dispatch[92]: y_load[92] = 305.4700012207031
 c_load_re-dispatch[93]: y_load[93] = 304.6600036621094
 c_load_re-dispatch[94]: y_load[94] = 305.8999938964844
 c_load_re-dispatch[95]: y_load[95] = 306
 c_y_cut[0]: x_cut[0] + y_cut[0] <= 0
 c_y_cut[1]: x_cut[1] + y_cut[1] <= 0
 c_y_cut[2]: x_cut[2] + y_cut[2] <= 0
 c_y_cut[3]: x_cut[3] + y_cut[3] <= 0
 c_y_cut[4]: x_cut[4] + y_cut[4] <= 0
 c_y_cut[5]: x_cut[5] + y_cut[5] <= 0
 c_y_cut[6]: x_cut[6] + y_cut[6] <= 0
 c_y_cut[7]: x_cut[7] + y_cut[7] <= 0
 c_y_cut[8]: x_cut[8] + y_cut[8] <= 0
 c_y_cut[9]: x_cut[9] + y_cut[9] <= 0
 c_y_cut[10]: x_cut[10] + y_cut[10] <= 0
 c_y_cut[11]: x_cut[11] + y_cut[11] <= 0
 c_y_cut[12]: x_cut[12] + y_cut[12] <= 0
 c_y_cut[13]: x_cut[13] + y_cut[13] <= 0
 c_y_cut[14]: x_cut[14] + y_cut[14] <= 0
 c_y_cut[15]: x_cut[15] + y_cut[15] <= 0
 c_y_cut[16]: x_cut[16] + y_cut[16] <= 0
 c_y_cut[17]: x_cut[17] + y_cut[17] <= 0
 c_y_cut[18]: x_cut[18] + y_cut[18] <= 0
 c_y_cut[19]: x_cut[19] + y_cut[19] <= 0
 c_y_cut[20]: x_cut[20] + y_cut[20] <= 0
 c_y_cut[21]: x_cut[21] + y_cut[21] <= 0
 c_y_cut[22]: x_cut[22] + y_cut[22] <= 0
 c_y_cut[23]: x_cut[23] + y_cut[23] <= 0
 c_y_cut[24]: x_cut[24] + y_cut[24] <= 0
 c_y_cut[25]: x_cut[25] + y_cut[25] <= 0
 c_y_cut[26]: x_cut[26] + y_cut[26] <= 0
 c_y_cut[27]: x_cut[27] + y_cut[27] <= 0
 c_y_cut[28]: x_cut[28] + y_cut[28] <= 0
 c_y_cut[29]: x_cut[29] + y_cut[29] <= 0.6817549108937482
 c_y_cut[30]: x_cut[30] + y_cut[30] <= 3.927929373725216
 c_y_cut[31]: x_cut[31] + y_cut[31] <= 12.01863468178648
 c_y_cut[32]: x_cut[32] + y_cut[32] <= 30.51313193721121
 c_y_cut[33]: x_cut[33] + y_cut[33] <= 35.08499965125498
 c_y_cut[34]: x_cut[34] + y_cut[34] <= 55.17332513737617
 c_y_cut[35]: x_cut[35] + y_cut[35] <= 140.1686278350507
 c_y_cut[36]: x_cut[36] + y_cut[36] <= 178.2836440688019
 c_y_cut[37]: x_cut[37] + y_cut[37] <= 207.9424740079004
 c_y_cut[38]: x_cut[38] + y_cut[38] <= 232.8479934620799
 c_y_cut[39]: x_cut[39] + y_cut[39] <= 263.3768905490792
 c_y_cut[40]: x_cut[40] + y_cut[40] <= 287.0985454425697
 c_y_cut[41]: x_cut[41] + y_cut[41] <= 309.8822349462725
 c_y_cut[42]: x_cut[42] + y_cut[42] <= 333.2777732532034
 c_y_cut[43]: x_cut[43] + y_cut[43] <= 352.389781719958
 c_y_cut[44]: x_cut[44] + y_cut[44] <= 375.7159161102198
 c_y_cut[45]: x_cut[45] + y_cut[45] <= 394.9875487973123
 c_y_cut[46]: x_cut[46] + y_cut[46] <= 409.7917531813302
 c_y_cut[47]: x_cut[47] + y_cut[47] <= 428.101815569561
 c_y_cut[48]: x_cut[48] + y_cut[48] <= 440.6464964382849
 c_y_cut[49]: x_cut[49] + y_cut[49] <= 445.3786620241449
 c_y_cut[50]: x_cut[50] + y_cut[50] <= 455.556472272189
 c_y_cut[51]: x_cut[51] + y_cut[51] <= 460.5117236642729
 c_y_cut[52]: x_cut[52] + y_cut[52] <= 464.00084206478
 c_y_cut[53]: x_cut[53] + y_cut[53] <= 469.2390762828775
 c_y_cut[54]: x_cut[54] + y_cut[54] <= 476.6880525993341
 c_y_cut[55]: x_cut[55] + y_cut[55] <= 473.7111903963382
 c_y_cut[56]: x_cut[56] + y_cut[56] <= 455.8462723276253
 c_y_cut[57]: x_cut[57] + y_cut[57] <= 436.6431681623882
 c_y_cut[58]: x_cut[58] + y_cut[58] <= 414.4684218590571
 c_y_cut[59]: x_cut[59] + y_cut[59] <= 393.7351348159873
 c_y_cut[60]: x_cut[60] + y_cut[60] <= 290.1728620584943
 c_y_cut[61]: x_cut[61] + y_cut[61] <= 350.6164410277992
 c_y_cut[62]: x_cut[62] + y_cut[62] <= 322.9022729394957
 c_y_cut[63]: x_cut[63] + y_cut[63] <= 293.0620625079054
 c_y_cut[64]: x_cut[64] + y_cut[64] <= 262.7833201991659
 c_y_cut[65]: x_cut[65] + y_cut[65] <= 231.385172051662
 c_y_cut[66]: x_cut[66] + y_cut[66] <= 199.9895743469131
 c_y_cut[67]: x_cut[67] + y_cut[67] <= 165.7915647075124
 c_y_cut[68]: x_cut[68] + y_cut[68] <= 132.5896733187977
 c_y_cut[69]: x_cut[69] + y_cut[69] <= 98.15859347428747
 c_y_cut[70]: x_cut[70] + y_cut[70] <= 40.61665929262848
 c_y_cut[71]: x_cut[71] + y_cut[71] <= 30.04096046923865
 c_y_cut[72]: x_cut[72] + y_cut[72] <= 16.20481073549318
 c_y_cut[73]: x_cut[73] + y_cut[73] <= 2.626139118086846
 c_y_cut[74]: x_cut[74] + y_cut[74] <= 0
 c_y_cut[75]: x_cut[75] + y_cut[75] <= 0
 c_y_cut[76]: x_cut[76] + y_cut[76] <= 0
 c_y_cut[77]: x_cut[77] + y_cut[77] <= 0
 c_y_cut[78]: x_cut[78] + y_cut[78] <= 0
 c_y_cut[79]: x_cut[79] + y_cut[79] <= 0
 c_y_cut[80]: x_cut[80] + y_cut[80] <= 0
 c_y_cut[81]: x_cut[81] + y_cut[81] <= 0
 c_y_cut[82]: x_cut[82] + y_cut[82] <= 0
 c_y_cut[83]: x_cut[83] + y_cut[83] <= 0
 c_y_cut[84]: x_cut[84] + y_cut[84] <= 0
 c_y_cut[85]: x_cut[85] + y_cut[85] <= 0
 c_y_cut[86]: x_cut[86] + y_cut[86] <= 0
 c_y_cut[87]: x_cut[87] + y_cut[87] <= 0
 c_y_cut[88]: x_cut[88] + y_cut[88] <= 0
 c_y_cut[89]: x_cut[89] + y_cut[89] <= 0
 c_y_cut[90]: x_cut[90] + y_cut[90] <= 0
 c_y_cut[91]: x_cut[91] + y_cut[91] <= 0
 c_y_cut[92]: x_cut[92] + y_cut[92] <= 0
 c_y_cut[93]: x_cut[93] + y_cut[93] <= 0
 c_y_cut[94]: x_cut[94] + y_cut[94] <= 0
 c_y_cut[95]: x_cut[95] + y_cut[95] <= 0
 c_y_add[0]: - x_cut[0] + y_add[0] <= 0
 c_y_add[1]: - x_cut[1] + y_add[1] <= 0
 c_y_add[2]: - x_cut[2] + y_add[2] <= 0
 c_y_add[3]: - x_cut[3] + y_add[3] <= 0
 c_y_add[4]: - x_cut[4] + y_add[4] <= 0
 c_y_add[5]: - x_cut[5] + y_add[5] <= 0
 c_y_add[6]: - x_cut[6] + y_add[6] <= 0
 c_y_add[7]: - x_cut[7] + y_add[7] <= 0
 c_y_add[8]: - x_cut[8] + y_add[8] <= 0
 c_y_add[9]: - x_cut[9] + y_add[9] <= 0
 c_y_add[10]: - x_cut[10] + y_add[10] <= 0
 c_y_add[11]: - x_cut[11] + y_add[11] <= 0
 c_y_add[12]: - x_cut[12] + y_add[12] <= 0
 c_y_add[13]: - x_cut[13] + y_add[13] <= 0
 c_y_add[14]: - x_cut[14] + y_add[14] <= 0
 c_y_add[15]: - x_cut[15] + y_add[15] <= 0
 c_y_add[16]: - x_cut[16] + y_add[16] <= 0
 c_y_add[17]: - x_cut[17] + y_add[17] <= 0
 c_y_add[18]: - x_cut[18] + y_add[18] <= 0
 c_y_add[19]: - x_cut[19] + y_add[19] <= 0
 c_y_add[20]: - x_cut[20] + y_add[20] <= 0
 c_y_add[21]: - x_cut[21] + y_add[21] <= 0
 c_y_add[22]: - x_cut[22] + y_add[22] <= 0
 c_y_add[23]: - x_cut[23] + y_add[23] <= 0
 c_y_add[24]: - x_cut[24] + y_add[24] <= 0
 c_y_add[25]: - x_cut[25] + y_add[25] <= 0
 c_y_add[26]: - x_cut[26] + y_add[26] <= 0
 c_y_add[27]: - x_cut[27] + y_add[27] <= 0
 c_y_add[28]: - x_cut[28] + y_add[28] <= 0
 c_y_add[29]: - x_cut[29] + y_add[29] <= 0
 c_y_add[30]: - x_cut[30] + y_add[30] <= 0
 c_y_add[31]: - x_cut[31] + y_add[31] <= 0
 c_y_add[32]: - x_cut[32] + y_add[32] <= 0
 c_y_add[33]: - x_cut[33] + y_add[33] <= 0
 c_y_add[34]: - x_cut[34] + y_add[34] <= 0
 c_y_add[35]: - x_cut[35] + y_add[35] <= 0
 c_y_add[36]: - x_cut[36] + y_add[36] <= 0
 c_y_add[37]: - x_cut[37] + y_add[37] <= 0
 c_y_add[38]: - x_cut[38] + y_add[38] <= 0
 c_y_add[39]: - x_cut[39] + y_add[39] <= 0
 c_y_add[40]: - x_cut[40] + y_add[40] <= 0
 c_y_add[41]: - x_cut[41] + y_add[41] <= 0
 c_y_add[42]: - x_cut[42] + y_add[42] <= 0
 c_y_add[43]: - x_cut[43] + y_add[43] <= 0
 c_y_add[44]: - x_cut[44] + y_add[44] <= 0
 c_y_add[45]: - x_cut[45] + y_add[45] <= 0
 c_y_add[46]: - x_cut[46] + y_add[46] <= 0
 c_y_add[47]: - x_cut[47] + y_add[47] <= 0
 c_y_add[48]: - x_cut[48] + y_add[48] <= 0
 c_y_add[49]: - x_cut[49] + y_add[49] <= 0
 c_y_add[50]: - x_cut[50] + y_add[50] <= 0
 c_y_add[51]: - x_cut[51] + y_add[51] <= 0
 c_y_add[52]: - x_cut[52] + y_add[52] <= 0
 c_y_add[53]: - x_cut[53] + y_add[53] <= 0
 c_y_add[54]: - x_cut[54] + y_add[54] <= 0
 c_y_add[55]: - x_cut[55] + y_add[55] <= 0
 c_y_add[56]: - x_cut[56] + y_add[56] <= 0
 c_y_add[57]: - x_cut[57] + y_add[57] <= 0
 c_y_add[58]: - x_cut[58] + y_add[58] <= 0
 c_y_add[59]: - x_cut[59] + y_add[59] <= 0
 c_y_add[60]: - x_cut[60] + y_add[60] <= 0
 c_y_add[61]: - x_cut[61] + y_add[61] <= 0
 c_y_add[62]: - x_cut[62] + y_add[62] <= 0
 c_y_add[63]: - x_cut[63] + y_add[63] <= 0
 c_y_add[64]: - x_cut[64] + y_add[64] <= 0
 c_y_add[65]: - x_cut[65] + y_add[65] <= 0
 c_y_add[66]: - x_cut[66] + y_add[66] <= 0
 c_y_add[67]: - x_cut[67] + y_add[67] <= 0
 c_y_add[68]: - x_cut[68] + y_add[68] <= 0
 c_y_add[69]: - x_cut[69] + y_add[69] <= 0
 c_y_add[70]: - x_cut[70] + y_add[70] <= 0
 c_y_add[71]: - x_cut[71] + y_add[71] <= 0
 c_y_add[72]: - x_cut[72] + y_add[72] <= 0
 c_y_add[73]: - x_cut[73] + y_add[73] <= 0
 c_y_add[74]: - x_cut[74] + y_add[74] <= 0
 c_y_add[75]: - x_cut[75] + y_add[75] <= 0
 c_y_add[76]: - x_cut[76] + y_add[76] <= 0
 c_y_add[77]: - x_cut[77] + y_add[77] <= 0
 c_y_add[78]: - x_cut[78] + y_add[78] <= 0
 c_y_add[79]: - x_cut[79] + y_add[79] <= 0
 c_y_add[80]: - x_cut[80] + y_add[80] <= 0
 c_y_add[81]: - x_cut[81] + y_add[81] <= 0
 c_y_add[82]: - x_cut[82] + y_add[82] <= 0
 c_y_add[83]: - x_cut[83] + y_add[83] <= 0
 c_y_add[84]: - x_cut[84] + y_add[84] <= 0
 c_y_add[85]: - x_cut[85] + y_add[85] <= 0
 c_y_add[86]: - x_cut[86] + y_add[86] <= 0
 c_y_add[87]: - x_cut[87] + y_add[87] <= 0
 c_y_add[88]: - x_cut[88] + y_add[88] <= 0
 c_y_add[89]: - x_cut[89] + y_add[89] <= 0
 c_y_add[90]: - x_cut[90] + y_add[90] <= 0
 c_y_add[91]: - x_cut[91] + y_add[91] <= 0
 c_y_add[92]: - x_cut[92] + y_add[92] <= 0
 c_y_add[93]: - x_cut[93] + y_add[93] <= 0
 c_y_add[94]: - x_cut[94] + y_add[94] <= 0
 c_y_add[95]: - x_cut[95] + y_add[95] <= 0
 c_cost_reg_DG[0]: - 0.2 y_pos[0] - 0.1 y_neg[0] + y_cost_fuel[0] = 0
 c_cost_reg_DG[1]: - 0.2 y_pos[1] - 0.1 y_neg[1] + y_cost_fuel[1] = 0
 c_cost_reg_DG[2]: - 0.2 y_pos[2] - 0.1 y_neg[2] + y_cost_fuel[2] = 0
 c_cost_reg_DG[3]: - 0.2 y_pos[3] - 0.1 y_neg[3] + y_cost_fuel[3] = 0
 c_cost_reg_DG[4]: - 0.2 y_pos[4] - 0.1 y_neg[4] + y_cost_fuel[4] = 0
 c_cost_reg_DG[5]: - 0.2 y_pos[5] - 0.1 y_neg[5] + y_cost_fuel[5] = 0
 c_cost_reg_DG[6]: - 0.2 y_pos[6] - 0.1 y_neg[6] + y_cost_fuel[6] = 0
 c_cost_reg_DG[7]: - 0.2 y_pos[7] - 0.1 y_neg[7] + y_cost_fuel[7] = 0
 c_cost_reg_DG[8]: - 0.2 y_pos[8] - 0.1 y_neg[8] + y_cost_fuel[8] = 0
 c_cost_reg_DG[9]: - 0.2 y_pos[9] - 0.1 y_neg[9] + y_cost_fuel[9] = 0
 c_cost_reg_DG[10]: - 0.2 y_pos[10] - 0.1 y_neg[10] + y_cost_fuel[10] = 0
 c_cost_reg_DG[11]: - 0.2 y_pos[11] - 0.1 y_neg[11] + y_cost_fuel[11] = 0
 c_cost_reg_DG[12]: - 0.2 y_pos[12] - 0.1 y_neg[12] + y_cost_fuel[12] = 0
 c_cost_reg_DG[13]: - 0.2 y_pos[13] - 0.1 y_neg[13] + y_cost_fuel[13] = 0
 c_cost_reg_DG[14]: - 0.2 y_pos[14] - 0.1 y_neg[14] + y_cost_fuel[14] = 0
 c_cost_reg_DG[15]: - 0.2 y_pos[15] - 0.1 y_neg[15] + y_cost_fuel[15] = 0
 c_cost_reg_DG[16]: - 0.2 y_pos[16] - 0.1 y_neg[16] + y_cost_fuel[16] = 0
 c_cost_reg_DG[17]: - 0.2 y_pos[17] - 0.1 y_neg[17] + y_cost_fuel[17] = 0
 c_cost_reg_DG[18]: - 0.2 y_pos[18] - 0.1 y_neg[18] + y_cost_fuel[18] = 0
 c_cost_reg_DG[19]: - 0.2 y_pos[19] - 0.1 y_neg[19] + y_cost_fuel[19] = 0
 c_cost_reg_DG[20]: - 0.2 y_pos[20] - 0.1 y_neg[20] + y_cost_fuel[20] = 0
 c_cost_reg_DG[21]: - 0.2 y_pos[21] - 0.1 y_neg[21] + y_cost_fuel[21] = 0
 c_cost_reg_DG[22]: - 0.2 y_pos[22] - 0.1 y_neg[22] + y_cost_fuel[22] = 0
 c_cost_reg_DG[23]: - 0.2 y_pos[23] - 0.1 y_neg[23] + y_cost_fuel[23] = 0
 c_cost_reg_DG[24]: - 0.2 y_pos[24] - 0.1 y_neg[24] + y_cost_fuel[24] = 0
 c_cost_reg_DG[25]: - 0.2 y_pos[25] - 0.1 y_neg[25] + y_cost_fuel[25] = 0
 c_cost_reg_DG[26]: - 0.2 y_pos[26] - 0.1 y_neg[26] + y_cost_fuel[26] = 0
 c_cost_reg_DG[27]: - 0.2 y_pos[27] - 0.1 y_neg[27] + y_cost_fuel[27] = 0
 c_cost_reg_DG[28]: - 0.2 y_pos[28] - 0.1 y_neg[28] + y_cost_fuel[28] = 0
 c_cost_reg_DG[29]: - 0.2 y_pos[29] - 0.1 y_neg[29] + y_cost_fuel[29] = 0
 c_cost_reg_DG[30]: - 0.2 y_pos[30] - 0.1 y_neg[30] + y_cost_fuel[30] = 0
 c_cost_reg_DG[31]: - 0.2 y_pos[31] - 0.1 y_neg[31] + y_cost_fuel[31] = 0
 c_cost_reg_DG[32]: - 0.2 y_pos[32] - 0.1 y_neg[32] + y_cost_fuel[32] = 0
 c_cost_reg_DG[33]: - 0.2 y_pos[33] - 0.1 y_neg[33] + y_cost_fuel[33] = 0
 c_cost_reg_DG[34]: - 0.2 y_pos[34] - 0.1 y_neg[34] + y_cost_fuel[34] = 0
 c_cost_reg_DG[35]: - 0.2 y_pos[35] - 0.1 y_neg[35] + y_cost_fuel[35] = 0
 c_cost_reg_DG[36]: - 0.2 y_pos[36] - 0.1 y_neg[36] + y_cost_fuel[36] = 0
 c_cost_reg_DG[37]: - 0.2 y_pos[37] - 0.1 y_neg[37] + y_cost_fuel[37] = 0
 c_cost_reg_DG[38]: - 0.2 y_pos[38] - 0.1 y_neg[38] + y_cost_fuel[38] = 0
 c_cost_reg_DG[39]: - 0.2 y_pos[39] - 0.1 y_neg[39] + y_cost_fuel[39] = 0
 c_cost_reg_DG[40]: - 0.2 y_pos[40] - 0.1 y_neg[40] + y_cost_fuel[40] = 0
 c_cost_reg_DG[41]: - 0.2 y_pos[41] - 0.1 y_neg[41] + y_cost_fuel[41] = 0
 c_cost_reg_DG[42]: - 0.2 y_pos[42] - 0.1 y_neg[42] + y_cost_fuel[42] = 0
 c_cost_reg_DG[43]: - 0.2 y_pos[43] - 0.1 y_neg[43] + y_cost_fuel[43] = 0
 c_cost_reg_DG[44]: - 0.2 y_pos[44] - 0.1 y_neg[44] + y_cost_fuel[44] = 0
 c_cost_reg_DG[45]: - 0.2 y_pos[45] - 0.1 y_neg[45] + y_cost_fuel[45] = 0
 c_cost_reg_DG[46]: - 0.2 y_pos[46] - 0.1 y_neg[46] + y_cost_fuel[46] = 0
 c_cost_reg_DG[47]: - 0.2 y_pos[47] - 0.1 y_neg[47] + y_cost_fuel[47] = 0
 c_cost_reg_DG[48]: - 0.2 y_pos[48] - 0.1 y_neg[48] + y_cost_fuel[48] = 0
 c_cost_reg_DG[49]: - 0.2 y_pos[49] - 0.1 y_neg[49] + y_cost_fuel[49] = 0
 c_cost_reg_DG[50]: - 0.2 y_pos[50] - 0.1 y_neg[50] + y_cost_fuel[50] = 0
 c_cost_reg_DG[51]: - 0.2 y_pos[51] - 0.1 y_neg[51] + y_cost_fuel[51] = 0
 c_cost_reg_DG[52]: - 0.2 y_pos[52] - 0.1 y_neg[52] + y_cost_fuel[52] = 0
 c_cost_reg_DG[53]: - 0.2 y_pos[53] - 0.1 y_neg[53] + y_cost_fuel[53] = 0
 c_cost_reg_DG[54]: - 0.2 y_pos[54] - 0.1 y_neg[54] + y_cost_fuel[54] = 0
 c_cost_reg_DG[55]: - 0.2 y_pos[55] - 0.1 y_neg[55] + y_cost_fuel[55] = 0
 c_cost_reg_DG[56]: - 0.2 y_pos[56] - 0.1 y_neg[56] + y_cost_fuel[56] = 0
 c_cost_reg_DG[57]: - 0.2 y_pos[57] - 0.1 y_neg[57] + y_cost_fuel[57] = 0
 c_cost_reg_DG[58]: - 0.2 y_pos[58] - 0.1 y_neg[58] + y_cost_fuel[58] = 0
 c_cost_reg_DG[59]: - 0.2 y_pos[59] - 0.1 y_neg[59] + y_cost_fuel[59] = 0
 c_cost_reg_DG[60]: - 0.2 y_pos[60] - 0.1 y_neg[60] + y_cost_fuel[60] = 0
 c_cost_reg_DG[61]: - 0.2 y_pos[61] - 0.1 y_neg[61] + y_cost_fuel[61] = 0
 c_cost_reg_DG[62]: - 0.2 y_pos[62] - 0.1 y_neg[62] + y_cost_fuel[62] = 0
 c_cost_reg_DG[63]: - 0.2 y_pos[63] - 0.1 y_neg[63] + y_cost_fuel[63] = 0
 c_cost_reg_DG[64]: - 0.2 y_pos[64] - 0.1 y_neg[64] + y_cost_fuel[64] = 0
 c_cost_reg_DG[65]: - 0.2 y_pos[65] - 0.1 y_neg[65] + y_cost_fuel[65] = 0
 c_cost_reg_DG[66]: - 0.2 y_pos[66] - 0.1 y_neg[66] + y_cost_fuel[66] = 0
 c_cost_reg_DG[67]: - 0.2 y_pos[67] - 0.1 y_neg[67] + y_cost_fuel[67] = 0
 c_cost_reg_DG[68]: - 0.2 y_pos[68] - 0.1 y_neg[68] + y_cost_fuel[68] = 0
 c_cost_reg_DG[69]: - 0.2 y_pos[69] - 0.1 y_neg[69] + y_cost_fuel[69] = 0
 c_cost_reg_DG[70]: - 0.2 y_pos[70] - 0.1 y_neg[70] + y_cost_fuel[70] = 0
 c_cost_reg_DG[71]: - 0.2 y_pos[71] - 0.1 y_neg[71] + y_cost_fuel[71] = 0
 c_cost_reg_DG[72]: - 0.2 y_pos[72] - 0.1 y_neg[72] + y_cost_fuel[72] = 0
 c_cost_reg_DG[73]: - 0.2 y_pos[73] - 0.1 y_neg[73] + y_cost_fuel[73] = 0
 c_cost_reg_DG[74]: - 0.2 y_pos[74] - 0.1 y_neg[74] + y_cost_fuel[74] = 0
 c_cost_reg_DG[75]: - 0.2 y_pos[75] - 0.1 y_neg[75] + y_cost_fuel[75] = 0
 c_cost_reg_DG[76]: - 0.2 y_pos[76] - 0.1 y_neg[76] + y_cost_fuel[76] = 0
 c_cost_reg_DG[77]: - 0.2 y_pos[77] - 0.1 y_neg[77] + y_cost_fuel[77] = 0
 c_cost_reg_DG[78]: - 0.2 y_pos[78] - 0.1 y_neg[78] + y_cost_fuel[78] = 0
 c_cost_reg_DG[79]: - 0.2 y_pos[79] - 0.1 y_neg[79] + y_cost_fuel[79] = 0
 c_cost_reg_DG[80]: - 0.2 y_pos[80] - 0.1 y_neg[80] + y_cost_fuel[80] = 0
 c_cost_reg_DG[81]: - 0.2 y_pos[81] - 0.1 y_neg[81] + y_cost_fuel[81] = 0
 c_cost_reg_DG[82]: - 0.2 y_pos[82] - 0.1 y_neg[82] + y_cost_fuel[82] = 0
 c_cost_reg_DG[83]: - 0.2 y_pos[83] - 0.1 y_neg[83] + y_cost_fuel[83] = 0
 c_cost_reg_DG[84]: - 0.2 y_pos[84] - 0.1 y_neg[84] + y_cost_fuel[84] = 0
 c_cost_reg_DG[85]: - 0.2 y_pos[85] - 0.1 y_neg[85] + y_cost_fuel[85] = 0
 c_cost_reg_DG[86]: - 0.2 y_pos[86] - 0.1 y_neg[86] + y_cost_fuel[86] = 0
 c_cost_reg_DG[87]: - 0.2 y_pos[87] - 0.1 y_neg[87] + y_cost_fuel[87] = 0
 c_cost_reg_DG[88]: - 0.2 y_pos[88] - 0.1 y_neg[88] + y_cost_fuel[88] = 0
 c_cost_reg_DG[89]: - 0.2 y_pos[89] - 0.1 y_neg[89] + y_cost_fuel[89] = 0
 c_cost_reg_DG[90]: - 0.2 y_pos[90] - 0.1 y_neg[90] + y_cost_fuel[90] = 0
 c_cost_reg_DG[91]: - 0.2 y_pos[91] - 0.1 y_neg[91] + y_cost_fuel[91] = 0
 c_cost_reg_DG[92]: - 0.2 y_pos[92] - 0.1 y_neg[92] + y_cost_fuel[92] = 0
 c_cost_reg_DG[93]: - 0.2 y_pos[93] - 0.1 y_neg[93] + y_cost_fuel[93] = 0
 c_cost_reg_DG[94]: - 0.2 y_pos[94] - 0.1 y_neg[94] + y_cost_fuel[94] = 0
 c_cost_reg_DG[95]: - 0.2 y_pos[95] - 0.1 y_neg[95] + y_cost_fuel[95] = 0
 c_cost_reg_ES[0]: - 0.03 y_chg[0] - 0.03 y_dis[0] + y_cost_ESS[0] = 0
 c_cost_reg_ES[1]: - 0.03 y_chg[1] - 0.03 y_dis[1] + y_cost_ESS[1] = 0
 c_cost_reg_ES[2]: - 0.03 y_chg[2] - 0.03 y_dis[2] + y_cost_ESS[2] = 0
 c_cost_reg_ES[3]: - 0.03 y_chg[3] - 0.03 y_dis[3] + y_cost_ESS[3] = 0
 c_cost_reg_ES[4]: - 0.03 y_chg[4] - 0.03 y_dis[4] + y_cost_ESS[4] = 0
 c_cost_reg_ES[5]: - 0.03 y_chg[5] - 0.03 y_dis[5] + y_cost_ESS[5] = 0
 c_cost_reg_ES[6]: - 0.03 y_chg[6] - 0.03 y_dis[6] + y_cost_ESS[6] = 0
 c_cost_reg_ES[7]: - 0.03 y_chg[7] - 0.03 y_dis[7] + y_cost_ESS[7] = 0
 c_cost_reg_ES[8]: - 0.03 y_chg[8] - 0.03 y_dis[8] + y_cost_ESS[8] = 0
 c_cost_reg_ES[9]: - 0.03 y_chg[9] - 0.03 y_dis[9] + y_cost_ESS[9] = 0
 c_cost_reg_ES[10]: - 0.03 y_chg[10] - 0.03 y_dis[10] + y_cost_ESS[10] = 0
 c_cost_reg_ES[11]: - 0.03 y_chg[11] - 0.03 y_dis[11] + y_cost_ESS[11] = 0
 c_cost_reg_ES[12]: - 0.03 y_chg[12] - 0.03 y_dis[12] + y_cost_ESS[12] = 0
 c_cost_reg_ES[13]: - 0.03 y_chg[13] - 0.03 y_dis[13] + y_cost_ESS[13] = 0
 c_cost_reg_ES[14]: - 0.03 y_chg[14] - 0.03 y_dis[14] + y_cost_ESS[14] = 0
 c_cost_reg_ES[15]: - 0.03 y_chg[15] - 0.03 y_dis[15] + y_cost_ESS[15] = 0
 c_cost_reg_ES[16]: - 0.03 y_chg[16] - 0.03 y_dis[16] + y_cost_ESS[16] = 0
 c_cost_reg_ES[17]: - 0.03 y_chg[17] - 0.03 y_dis[17] + y_cost_ESS[17] = 0
 c_cost_reg_ES[18]: - 0.03 y_chg[18] - 0.03 y_dis[18] + y_cost_ESS[18] = 0
 c_cost_reg_ES[19]: - 0.03 y_chg[19] - 0.03 y_dis[19] + y_cost_ESS[19] = 0
 c_cost_reg_ES[20]: - 0.03 y_chg[20] - 0.03 y_dis[20] + y_cost_ESS[20] = 0
 c_cost_reg_ES[21]: - 0.03 y_chg[21] - 0.03 y_dis[21] + y_cost_ESS[21] = 0
 c_cost_reg_ES[22]: - 0.03 y_chg[22] - 0.03 y_dis[22] + y_cost_ESS[22] = 0
 c_cost_reg_ES[23]: - 0.03 y_chg[23] - 0.03 y_dis[23] + y_cost_ESS[23] = 0
 c_cost_reg_ES[24]: - 0.03 y_chg[24] - 0.03 y_dis[24] + y_cost_ESS[24] = 0
 c_cost_reg_ES[25]: - 0.03 y_chg[25] - 0.03 y_dis[25] + y_cost_ESS[25] = 0
 c_cost_reg_ES[26]: - 0.03 y_chg[26] - 0.03 y_dis[26] + y_cost_ESS[26] = 0
 c_cost_reg_ES[27]: - 0.03 y_chg[27] - 0.03 y_dis[27] + y_cost_ESS[27] = 0
 c_cost_reg_ES[28]: - 0.03 y_chg[28] - 0.03 y_dis[28] + y_cost_ESS[28] = 0
 c_cost_reg_ES[29]: - 0.03 y_chg[29] - 0.03 y_dis[29] + y_cost_ESS[29] = 0
 c_cost_reg_ES[30]: - 0.03 y_chg[30] - 0.03 y_dis[30] + y_cost_ESS[30] = 0
 c_cost_reg_ES[31]: - 0.03 y_chg[31] - 0.03 y_dis[31] + y_cost_ESS[31] = 0
 c_cost_reg_ES[32]: - 0.03 y_chg[32] - 0.03 y_dis[32] + y_cost_ESS[32] = 0
 c_cost_reg_ES[33]: - 0.03 y_chg[33] - 0.03 y_dis[33] + y_cost_ESS[33] = 0
 c_cost_reg_ES[34]: - 0.03 y_chg[34] - 0.03 y_dis[34] + y_cost_ESS[34] = 0
 c_cost_reg_ES[35]: - 0.03 y_chg[35] - 0.03 y_dis[35] + y_cost_ESS[35] = 0
 c_cost_reg_ES[36]: - 0.03 y_chg[36] - 0.03 y_dis[36] + y_cost_ESS[36] = 0
 c_cost_reg_ES[37]: - 0.03 y_chg[37] - 0.03 y_dis[37] + y_cost_ESS[37] = 0
 c_cost_reg_ES[38]: - 0.03 y_chg[38] - 0.03 y_dis[38] + y_cost_ESS[38] = 0
 c_cost_reg_ES[39]: - 0.03 y_chg[39] - 0.03 y_dis[39] + y_cost_ESS[39] = 0
 c_cost_reg_ES[40]: - 0.03 y_chg[40] - 0.03 y_dis[40] + y_cost_ESS[40] = 0
 c_cost_reg_ES[41]: - 0.03 y_chg[41] - 0.03 y_dis[41] + y_cost_ESS[41] = 0
 c_cost_reg_ES[42]: - 0.03 y_chg[42] - 0.03 y_dis[42] + y_cost_ESS[42] = 0
 c_cost_reg_ES[43]: - 0.03 y_chg[43] - 0.03 y_dis[43] + y_cost_ESS[43] = 0
 c_cost_reg_ES[44]: - 0.03 y_chg[44] - 0.03 y_dis[44] + y_cost_ESS[44] = 0
 c_cost_reg_ES[45]: - 0.03 y_chg[45] - 0.03 y_dis[45] + y_cost_ESS[45] = 0
 c_cost_reg_ES[46]: - 0.03 y_chg[46] - 0.03 y_dis[46] + y_cost_ESS[46] = 0
 c_cost_reg_ES[47]: - 0.03 y_chg[47] - 0.03 y_dis[47] + y_cost_ESS[47] = 0
 c_cost_reg_ES[48]: - 0.03 y_chg[48] - 0.03 y_dis[48] + y_cost_ESS[48] = 0
 c_cost_reg_ES[49]: - 0.03 y_chg[49] - 0.03 y_dis[49] + y_cost_ESS[49] = 0
 c_cost_reg_ES[50]: - 0.03 y_chg[50] - 0.03 y_dis[50] + y_cost_ESS[50] = 0
 c_cost_reg_ES[51]: - 0.03 y_chg[51] - 0.03 y_dis[51] + y_cost_ESS[51] = 0
 c_cost_reg_ES[52]: - 0.03 y_chg[52] - 0.03 y_dis[52] + y_cost_ESS[52] = 0
 c_cost_reg_ES[53]: - 0.03 y_chg[53] - 0.03 y_dis[53] + y_cost_ESS[53] = 0
 c_cost_reg_ES[54]: - 0.03 y_chg[54] - 0.03 y_dis[54] + y_cost_ESS[54] = 0
 c_cost_reg_ES[55]: - 0.03 y_chg[55] - 0.03 y_dis[55] + y_cost_ESS[55] = 0
 c_cost_reg_ES[56]: - 0.03 y_chg[56] - 0.03 y_dis[56] + y_cost_ESS[56] = 0
 c_cost_reg_ES[57]: - 0.03 y_chg[57] - 0.03 y_dis[57] + y_cost_ESS[57] = 0
 c_cost_reg_ES[58]: - 0.03 y_chg[58] - 0.03 y_dis[58] + y_cost_ESS[58] = 0
 c_cost_reg_ES[59]: - 0.03 y_chg[59] - 0.03 y_dis[59] + y_cost_ESS[59] = 0
 c_cost_reg_ES[60]: - 0.03 y_chg[60] - 0.03 y_dis[60] + y_cost_ESS[60] = 0
 c_cost_reg_ES[61]: - 0.03 y_chg[61] - 0.03 y_dis[61] + y_cost_ESS[61] = 0
 c_cost_reg_ES[62]: - 0.03 y_chg[62] - 0.03 y_dis[62] + y_cost_ESS[62] = 0
 c_cost_reg_ES[63]: - 0.03 y_chg[63] - 0.03 y_dis[63] + y_cost_ESS[63] = 0
 c_cost_reg_ES[64]: - 0.03 y_chg[64] - 0.03 y_dis[64] + y_cost_ESS[64] = 0
 c_cost_reg_ES[65]: - 0.03 y_chg[65] - 0.03 y_dis[65] + y_cost_ESS[65] = 0
 c_cost_reg_ES[66]: - 0.03 y_chg[66] - 0.03 y_dis[66] + y_cost_ESS[66] = 0
 c_cost_reg_ES[67]: - 0.03 y_chg[67] - 0.03 y_dis[67] + y_cost_ESS[67] = 0
 c_cost_reg_ES[68]: - 0.03 y_chg[68] - 0.03 y_dis[68] + y_cost_ESS[68] = 0
 c_cost_reg_ES[69]: - 0.03 y_chg[69] - 0.03 y_dis[69] + y_cost_ESS[69] = 0
 c_cost_reg_ES[70]: - 0.03 y_chg[70] - 0.03 y_dis[70] + y_cost_ESS[70] = 0
 c_cost_reg_ES[71]: - 0.03 y_chg[71] - 0.03 y_dis[71] + y_cost_ESS[71] = 0
 c_cost_reg_ES[72]: - 0.03 y_chg[72] - 0.03 y_dis[72] + y_cost_ESS[72] = 0
 c_cost_reg_ES[73]: - 0.03 y_chg[73] - 0.03 y_dis[73] + y_cost_ESS[73] = 0
 c_cost_reg_ES[74]: - 0.03 y_chg[74] - 0.03 y_dis[74] + y_cost_ESS[74] = 0
 c_cost_reg_ES[75]: - 0.03 y_chg[75] - 0.03 y_dis[75] + y_cost_ESS[75] = 0
 c_cost_reg_ES[76]: - 0.03 y_chg[76] - 0.03 y_dis[76] + y_cost_ESS[76] = 0
 c_cost_reg_ES[77]: - 0.03 y_chg[77] - 0.03 y_dis[77] + y_cost_ESS[77] = 0
 c_cost_reg_ES[78]: - 0.03 y_chg[78] - 0.03 y_dis[78] + y_cost_ESS[78] = 0
 c_cost_reg_ES[79]: - 0.03 y_chg[79] - 0.03 y_dis[79] + y_cost_ESS[79] = 0
 c_cost_reg_ES[80]: - 0.03 y_chg[80] - 0.03 y_dis[80] + y_cost_ESS[80] = 0
 c_cost_reg_ES[81]: - 0.03 y_chg[81] - 0.03 y_dis[81] + y_cost_ESS[81] = 0
 c_cost_reg_ES[82]: - 0.03 y_chg[82] - 0.03 y_dis[82] + y_cost_ESS[82] = 0
 c_cost_reg_ES[83]: - 0.03 y_chg[83] - 0.03 y_dis[83] + y_cost_ESS[83] = 0
 c_cost_reg_ES[84]: - 0.03 y_chg[84] - 0.03 y_dis[84] + y_cost_ESS[84] = 0
 c_cost_reg_ES[85]: - 0.03 y_chg[85] - 0.03 y_dis[85] + y_cost_ESS[85] = 0
 c_cost_reg_ES[86]: - 0.03 y_chg[86] - 0.03 y_dis[86] + y_cost_ESS[86] = 0
 c_cost_reg_ES[87]: - 0.03 y_chg[87] - 0.03 y_dis[87] + y_cost_ESS[87] = 0
 c_cost_reg_ES[88]: - 0.03 y_chg[88] - 0.03 y_dis[88] + y_cost_ESS[88] = 0
 c_cost_reg_ES[89]: - 0.03 y_chg[89] - 0.03 y_dis[89] + y_cost_ESS[89] = 0
 c_cost_reg_ES[90]: - 0.03 y_chg[90] - 0.03 y_dis[90] + y_cost_ESS[90] = 0
 c_cost_reg_ES[91]: - 0.03 y_chg[91] - 0.03 y_dis[91] + y_cost_ESS[91] = 0
 c_cost_reg_ES[92]: - 0.03 y_chg[92] - 0.03 y_dis[92] + y_cost_ESS[92] = 0
 c_cost_reg_ES[93]: - 0.03 y_chg[93] - 0.03 y_dis[93] + y_cost_ESS[93] = 0
 c_cost_reg_ES[94]: - 0.03 y_chg[94] - 0.03 y_dis[94] + y_cost_ESS[94] = 0
 c_cost_reg_ES[95]: - 0.03 y_chg[95] - 0.03 y_dis[95] + y_cost_ESS[95] = 0
 c_cost_PV_cut_re[0]: - 0.04 y_cut[0] + y_cost_cut[0] = 0
 c_cost_PV_cut_re[1]: - 0.04 y_cut[1] + y_cost_cut[1] = 0
 c_cost_PV_cut_re[2]: - 0.04 y_cut[2] + y_cost_cut[2] = 0
 c_cost_PV_cut_re[3]: - 0.04 y_cut[3] + y_cost_cut[3] = 0
 c_cost_PV_cut_re[4]: - 0.04 y_cut[4] + y_cost_cut[4] = 0
 c_cost_PV_cut_re[5]: - 0.04 y_cut[5] + y_cost_cut[5] = 0
 c_cost_PV_cut_re[6]: - 0.04 y_cut[6] + y_cost_cut[6] = 0
 c_cost_PV_cut_re[7]: - 0.04 y_cut[7] + y_cost_cut[7] = 0
 c_cost_PV_cut_re[8]: - 0.04 y_cut[8] + y_cost_cut[8] = 0
 c_cost_PV_cut_re[9]: - 0.04 y_cut[9] + y_cost_cut[9] = 0
 c_cost_PV_cut_re[10]: - 0.04 y_cut[10] + y_cost_cut[10] = 0
 c_cost_PV_cut_re[11]: - 0.04 y_cut[11] + y_cost_cut[11] = 0
 c_cost_PV_cut_re[12]: - 0.04 y_cut[12] + y_cost_cut[12] = 0
 c_cost_PV_cut_re[13]: - 0.04 y_cut[13] + y_cost_cut[13] = 0
 c_cost_PV_cut_re[14]: - 0.04 y_cut[14] + y_cost_cut[14] = 0
 c_cost_PV_cut_re[15]: - 0.04 y_cut[15] + y_cost_cut[15] = 0
 c_cost_PV_cut_re[16]: - 0.04 y_cut[16] + y_cost_cut[16] = 0
 c_cost_PV_cut_re[17]: - 0.04 y_cut[17] + y_cost_cut[17] = 0
 c_cost_PV_cut_re[18]: - 0.04 y_cut[18] + y_cost_cut[18] = 0
 c_cost_PV_cut_re[19]: - 0.04 y_cut[19] + y_cost_cut[19] = 0
 c_cost_PV_cut_re[20]: - 0.04 y_cut[20] + y_cost_cut[20] = 0
 c_cost_PV_cut_re[21]: - 0.04 y_cut[21] + y_cost_cut[21] = 0
 c_cost_PV_cut_re[22]: - 0.04 y_cut[22] + y_cost_cut[22] = 0
 c_cost_PV_cut_re[23]: - 0.04 y_cut[23] + y_cost_cut[23] = 0
 c_cost_PV_cut_re[24]: - 0.04 y_cut[24] + y_cost_cut[24] = 0
 c_cost_PV_cut_re[25]: - 0.04 y_cut[25] + y_cost_cut[25] = 0
 c_cost_PV_cut_re[26]: - 0.04 y_cut[26] + y_cost_cut[26] = 0
 c_cost_PV_cut_re[27]: - 0.04 y_cut[27] + y_cost_cut[27] = 0
 c_cost_PV_cut_re[28]: - 0.04 y_cut[28] + y_cost_cut[28] = 0
 c_cost_PV_cut_re[29]: - 0.04 y_cut[29] + y_cost_cut[29] = 0
 c_cost_PV_cut_re[30]: - 0.04 y_cut[30] + y_cost_cut[30] = 0
 c_cost_PV_cut_re[31]: - 0.04 y_cut[31] + y_cost_cut[31] = 0
 c_cost_PV_cut_re[32]: - 0.04 y_cut[32] + y_cost_cut[32] = 0
 c_cost_PV_cut_re[33]: - 0.04 y_cut[33] + y_cost_cut[33] = 0
 c_cost_PV_cut_re[34]: - 0.04 y_cut[34] + y_cost_cut[34] = 0
 c_cost_PV_cut_re[35]: - 0.04 y_cut[35] + y_cost_cut[35] = 0
 c_cost_PV_cut_re[36]: - 0.04 y_cut[36] + y_cost_cut[36] = 0
 c_cost_PV_cut_re[37]: - 0.04 y_cut[37] + y_cost_cut[37] = 0
 c_cost_PV_cut_re[38]: - 0.04 y_cut[38] + y_cost_cut[38] = 0
 c_cost_PV_cut_re[39]: - 0.04 y_cut[39] + y_cost_cut[39] = 0
 c_cost_PV_cut_re[40]: - 0.04 y_cut[40] + y_cost_cut[40] = 0
 c_cost_PV_cut_re[41]: - 0.04 y_cut[41] + y_cost_cut[41] = 0
 c_cost_PV_cut_re[42]: - 0.04 y_cut[42] + y_cost_cut[42] = 0
 c_cost_PV_cut_re[43]: - 0.04 y_cut[43] + y_cost_cut[43] = 0
 c_cost_PV_cut_re[44]: - 0.04 y_cut[44] + y_cost_cut[44] = 0
 c_cost_PV_cut_re[45]: - 0.04 y_cut[45] + y_cost_cut[45] = 0
 c_cost_PV_cut_re[46]: - 0.04 y_cut[46] + y_cost_cut[46] = 0
 c_cost_PV_cut_re[47]: - 0.04 y_cut[47] + y_cost_cut[47] = 0
 c_cost_PV_cut_re[48]: - 0.04 y_cut[48] + y_cost_cut[48] = 0
 c_cost_PV_cut_re[49]: - 0.04 y_cut[49] + y_cost_cut[49] = 0
 c_cost_PV_cut_re[50]: - 0.04 y_cut[50] + y_cost_cut[50] = 0
 c_cost_PV_cut_re[51]: - 0.04 y_cut[51] + y_cost_cut[51] = 0
 c_cost_PV_cut_re[52]: - 0.04 y_cut[52] + y_cost_cut[52] = 0
 c_cost_PV_cut_re[53]: - 0.04 y_cut[53] + y_cost_cut[53] = 0
 c_cost_PV_cut_re[54]: - 0.04 y_cut[54] + y_cost_cut[54] = 0
 c_cost_PV_cut_re[55]: - 0.04 y_cut[55] + y_cost_cut[55] = 0
 c_cost_PV_cut_re[56]: - 0.04 y_cut[56] + y_cost_cut[56] = 0
 c_cost_PV_cut_re[57]: - 0.04 y_cut[57] + y_cost_cut[57] = 0
 c_cost_PV_cut_re[58]: - 0.04 y_cut[58] + y_cost_cut[58] = 0
 c_cost_PV_cut_re[59]: - 0.04 y_cut[59] + y_cost_cut[59] = 0
 c_cost_PV_cut_re[60]: - 0.04 y_cut[60] + y_cost_cut[60] = 0
 c_cost_PV_cut_re[61]: - 0.04 y_cut[61] + y_cost_cut[61] = 0
 c_cost_PV_cut_re[62]: - 0.04 y_cut[62] + y_cost_cut[62] = 0
 c_cost_PV_cut_re[63]: - 0.04 y_cut[63] + y_cost_cut[63] = 0
 c_cost_PV_cut_re[64]: - 0.04 y_cut[64] + y_cost_cut[64] = 0
 c_cost_PV_cut_re[65]: - 0.04 y_cut[65] + y_cost_cut[65] = 0
 c_cost_PV_cut_re[66]: - 0.04 y_cut[66] + y_cost_cut[66] = 0
 c_cost_PV_cut_re[67]: - 0.04 y_cut[67] + y_cost_cut[67] = 0
 c_cost_PV_cut_re[68]: - 0.04 y_cut[68] + y_cost_cut[68] = 0
 c_cost_PV_cut_re[69]: - 0.04 y_cut[69] + y_cost_cut[69] = 0
 c_cost_PV_cut_re[70]: - 0.04 y_cut[70] + y_cost_cut[70] = 0
 c_cost_PV_cut_re[71]: - 0.04 y_cut[71] + y_cost_cut[71] = 0
 c_cost_PV_cut_re[72]: - 0.04 y_cut[72] + y_cost_cut[72] = 0
 c_cost_PV_cut_re[73]: - 0.04 y_cut[73] + y_cost_cut[73] = 0
 c_cost_PV_cut_re[74]: - 0.04 y_cut[74] + y_cost_cut[74] = 0
 c_cost_PV_cut_re[75]: - 0.04 y_cut[75] + y_cost_cut[75] = 0
 c_cost_PV_cut_re[76]: - 0.04 y_cut[76] + y_cost_cut[76] = 0
 c_cost_PV_cut_re[77]: - 0.04 y_cut[77] + y_cost_cut[77] = 0
 c_cost_PV_cut_re[78]: - 0.04 y_cut[78] + y_cost_cut[78] = 0
 c_cost_PV_cut_re[79]: - 0.04 y_cut[79] + y_cost_cut[79] = 0
 c_cost_PV_cut_re[80]: - 0.04 y_cut[80] + y_cost_cut[80] = 0
 c_cost_PV_cut_re[81]: - 0.04 y_cut[81] + y_cost_cut[81] = 0
 c_cost_PV_cut_re[82]: - 0.04 y_cut[82] + y_cost_cut[82] = 0
 c_cost_PV_cut_re[83]: - 0.04 y_cut[83] + y_cost_cut[83] = 0
 c_cost_PV_cut_re[84]: - 0.04 y_cut[84] + y_cost_cut[84] = 0
 c_cost_PV_cut_re[85]: - 0.04 y_cut[85] + y_cost_cut[85] = 0
 c_cost_PV_cut_re[86]: - 0.04 y_cut[86] + y_cost_cut[86] = 0
 c_cost_PV_cut_re[87]: - 0.04 y_cut[87] + y_cost_cut[87] = 0
 c_cost_PV_cut_re[88]: - 0.04 y_cut[88] + y_cost_cut[88] = 0
 c_cost_PV_cut_re[89]: - 0.04 y_cut[89] + y_cost_cut[89] = 0
 c_cost_PV_cut_re[90]: - 0.04 y_cut[90] + y_cost_cut[90] = 0
 c_cost_PV_cut_re[91]: - 0.04 y_cut[91] + y_cost_cut[91] = 0
 c_cost_PV_cut_re[92]: - 0.04 y_cut[92] + y_cost_cut[92] = 0
 c_cost_PV_cut_re[93]: - 0.04 y_cut[93] + y_cost_cut[93] = 0
 c_cost_PV_cut_re[94]: - 0.04 y_cut[94] + y_cost_cut[94] = 0
 c_cost_PV_cut_re[95]: - 0.04 y_cut[95] + y_cost_cut[95] = 0
 c_cost_PV_add_re[0]: - 0.01 y_add[0] + y_cost_add[0] = 0
 c_cost_PV_add_re[1]: - 0.01 y_add[1] + y_cost_add[1] = 0
 c_cost_PV_add_re[2]: - 0.01 y_add[2] + y_cost_add[2] = 0
 c_cost_PV_add_re[3]: - 0.01 y_add[3] + y_cost_add[3] = 0
 c_cost_PV_add_re[4]: - 0.01 y_add[4] + y_cost_add[4] = 0
 c_cost_PV_add_re[5]: - 0.01 y_add[5] + y_cost_add[5] = 0
 c_cost_PV_add_re[6]: - 0.01 y_add[6] + y_cost_add[6] = 0
 c_cost_PV_add_re[7]: - 0.01 y_add[7] + y_cost_add[7] = 0
 c_cost_PV_add_re[8]: - 0.01 y_add[8] + y_cost_add[8] = 0
 c_cost_PV_add_re[9]: - 0.01 y_add[9] + y_cost_add[9] = 0
 c_cost_PV_add_re[10]: - 0.01 y_add[10] + y_cost_add[10] = 0
 c_cost_PV_add_re[11]: - 0.01 y_add[11] + y_cost_add[11] = 0
 c_cost_PV_add_re[12]: - 0.01 y_add[12] + y_cost_add[12] = 0
 c_cost_PV_add_re[13]: - 0.01 y_add[13] + y_cost_add[13] = 0
 c_cost_PV_add_re[14]: - 0.01 y_add[14] + y_cost_add[14] = 0
 c_cost_PV_add_re[15]: - 0.01 y_add[15] + y_cost_add[15] = 0
 c_cost_PV_add_re[16]: - 0.01 y_add[16] + y_cost_add[16] = 0
 c_cost_PV_add_re[17]: - 0.01 y_add[17] + y_cost_add[17] = 0
 c_cost_PV_add_re[18]: - 0.01 y_add[18] + y_cost_add[18] = 0
 c_cost_PV_add_re[19]: - 0.01 y_add[19] + y_cost_add[19] = 0
 c_cost_PV_add_re[20]: - 0.01 y_add[20] + y_cost_add[20] = 0
 c_cost_PV_add_re[21]: - 0.01 y_add[21] + y_cost_add[21] = 0
 c_cost_PV_add_re[22]: - 0.01 y_add[22] + y_cost_add[22] = 0
 c_cost_PV_add_re[23]: - 0.01 y_add[23] + y_cost_add[23] = 0
 c_cost_PV_add_re[24]: - 0.01 y_add[24] + y_cost_add[24] = 0
 c_cost_PV_add_re[25]: - 0.01 y_add[25] + y_cost_add[25] = 0
 c_cost_PV_add_re[26]: - 0.01 y_add[26] + y_cost_add[26] = 0
 c_cost_PV_add_re[27]: - 0.01 y_add[27] + y_cost_add[27] = 0
 c_cost_PV_add_re[28]: - 0.01 y_add[28] + y_cost_add[28] = 0
 c_cost_PV_add_re[29]: - 0.01 y_add[29] + y_cost_add[29] = 0
 c_cost_PV_add_re[30]: - 0.01 y_add[30] + y_cost_add[30] = 0
 c_cost_PV_add_re[31]: - 0.01 y_add[31] + y_cost_add[31] = 0
 c_cost_PV_add_re[32]: - 0.01 y_add[32] + y_cost_add[32] = 0
 c_cost_PV_add_re[33]: - 0.01 y_add[33] + y_cost_add[33] = 0
 c_cost_PV_add_re[34]: - 0.01 y_add[34] + y_cost_add[34] = 0
 c_cost_PV_add_re[35]: - 0.01 y_add[35] + y_cost_add[35] = 0
 c_cost_PV_add_re[36]: - 0.01 y_add[36] + y_cost_add[36] = 0
 c_cost_PV_add_re[37]: - 0.01 y_add[37] + y_cost_add[37] = 0
 c_cost_PV_add_re[38]: - 0.01 y_add[38] + y_cost_add[38] = 0
 c_cost_PV_add_re[39]: - 0.01 y_add[39] + y_cost_add[39] = 0
 c_cost_PV_add_re[40]: - 0.01 y_add[40] + y_cost_add[40] = 0
 c_cost_PV_add_re[41]: - 0.01 y_add[41] + y_cost_add[41] = 0
 c_cost_PV_add_re[42]: - 0.01 y_add[42] + y_cost_add[42] = 0
 c_cost_PV_add_re[43]: - 0.01 y_add[43] + y_cost_add[43] = 0
 c_cost_PV_add_re[44]: - 0.01 y_add[44] + y_cost_add[44] = 0
 c_cost_PV_add_re[45]: - 0.01 y_add[45] + y_cost_add[45] = 0
 c_cost_PV_add_re[46]: - 0.01 y_add[46] + y_cost_add[46] = 0
 c_cost_PV_add_re[47]: - 0.01 y_add[47] + y_cost_add[47] = 0
 c_cost_PV_add_re[48]: - 0.01 y_add[48] + y_cost_add[48] = 0
 c_cost_PV_add_re[49]: - 0.01 y_add[49] + y_cost_add[49] = 0
 c_cost_PV_add_re[50]: - 0.01 y_add[50] + y_cost_add[50] = 0
 c_cost_PV_add_re[51]: - 0.01 y_add[51] + y_cost_add[51] = 0
 c_cost_PV_add_re[52]: - 0.01 y_add[52] + y_cost_add[52] = 0
 c_cost_PV_add_re[53]: - 0.01 y_add[53] + y_cost_add[53] = 0
 c_cost_PV_add_re[54]: - 0.01 y_add[54] + y_cost_add[54] = 0
 c_cost_PV_add_re[55]: - 0.01 y_add[55] + y_cost_add[55] = 0
 c_cost_PV_add_re[56]: - 0.01 y_add[56] + y_cost_add[56] = 0
 c_cost_PV_add_re[57]: - 0.01 y_add[57] + y_cost_add[57] = 0
 c_cost_PV_add_re[58]: - 0.01 y_add[58] + y_cost_add[58] = 0
 c_cost_PV_add_re[59]: - 0.01 y_add[59] + y_cost_add[59] = 0
 c_cost_PV_add_re[60]: - 0.01 y_add[60] + y_cost_add[60] = 0
 c_cost_PV_add_re[61]: - 0.01 y_add[61] + y_cost_add[61] = 0
 c_cost_PV_add_re[62]: - 0.01 y_add[62] + y_cost_add[62] = 0
 c_cost_PV_add_re[63]: - 0.01 y_add[63] + y_cost_add[63] = 0
 c_cost_PV_add_re[64]: - 0.01 y_add[64] + y_cost_add[64] = 0
 c_cost_PV_add_re[65]: - 0.01 y_add[65] + y_cost_add[65] = 0
 c_cost_PV_add_re[66]: - 0.01 y_add[66] + y_cost_add[66] = 0
 c_cost_PV_add_re[67]: - 0.01 y_add[67] + y_cost_add[67] = 0
 c_cost_PV_add_re[68]: - 0.01 y_add[68] + y_cost_add[68] = 0
 c_cost_PV_add_re[69]: - 0.01 y_add[69] + y_cost_add[69] = 0
 c_cost_PV_add_re[70]: - 0.01 y_add[70] + y_cost_add[70] = 0
 c_cost_PV_add_re[71]: - 0.01 y_add[71] + y_cost_add[71] = 0
 c_cost_PV_add_re[72]: - 0.01 y_add[72] + y_cost_add[72] = 0
 c_cost_PV_add_re[73]: - 0.01 y_add[73] + y_cost_add[73] = 0
 c_cost_PV_add_re[74]: - 0.01 y_add[74] + y_cost_add[74] = 0
 c_cost_PV_add_re[75]: - 0.01 y_add[75] + y_cost_add[75] = 0
 c_cost_PV_add_re[76]: - 0.01 y_add[76] + y_cost_add[76] = 0
 c_cost_PV_add_re[77]: - 0.01 y_add[77] + y_cost_add[77] = 0
 c_cost_PV_add_re[78]: - 0.01 y_add[78] + y_cost_add[78] = 0
 c_cost_PV_add_re[79]: - 0.01 y_add[79] + y_cost_add[79] = 0
 c_cost_PV_add_re[80]: - 0.01 y_add[80] + y_cost_add[80] = 0
 c_cost_PV_add_re[81]: - 0.01 y_add[81] + y_cost_add[81] = 0
 c_cost_PV_add_re[82]: - 0.01 y_add[82] + y_cost_add[82] = 0
 c_cost_PV_add_re[83]: - 0.01 y_add[83] + y_cost_add[83] = 0
 c_cost_PV_add_re[84]: - 0.01 y_add[84] + y_cost_add[84] = 0
 c_cost_PV_add_re[85]: - 0.01 y_add[85] + y_cost_add[85] = 0
 c_cost_PV_add_re[86]: - 0.01 y_add[86] + y_cost_add[86] = 0
 c_cost_PV_add_re[87]: - 0.01 y_add[87] + y_cost_add[87] = 0
 c_cost_PV_add_re[88]: - 0.01 y_add[88] + y_cost_add[88] = 0
 c_cost_PV_add_re[89]: - 0.01 y_add[89] + y_cost_add[89] = 0
 c_cost_PV_add_re[90]: - 0.01 y_add[90] + y_cost_add[90] = 0
 c_cost_PV_add_re[91]: - 0.01 y_add[91] + y_cost_add[91] = 0
 c_cost_PV_add_re[92]: - 0.01 y_add[92] + y_cost_add[92] = 0
 c_cost_PV_add_re[93]: - 0.01 y_add[93] + y_cost_add[93] = 0
 c_cost_PV_add_re[94]: - 0.01 y_add[94] + y_cost_add[94] = 0
 c_cost_PV_add_re[95]: - 0.01 y_add[95] + y_cost_add[95] = 0
 R3456: x[0] - x_chg[0] + x_dis[0] - x_cut[0] + y_pos[0] - y_neg[0]
   - y_chg[0] + y_dis[0] + y_PV[0] - y_cut[0] + y_add[0] - y_load[0] = 0
 R3457: x[1] - x_chg[1] + x_dis[1] - x_cut[1] + y_pos[1] - y_neg[1]
   - y_chg[1] + y_dis[1] + y_PV[1] - y_cut[1] + y_add[1] - y_load[1] = 0
 R3458: x[2] - x_chg[2] + x_dis[2] - x_cut[2] + y_pos[2] - y_neg[2]
   - y_chg[2] + y_dis[2] + y_PV[2] - y_cut[2] + y_add[2] - y_load[2] = 0
 R3459: x[3] - x_chg[3] + x_dis[3] - x_cut[3] + y_pos[3] - y_neg[3]
   - y_chg[3] + y_dis[3] + y_PV[3] - y_cut[3] + y_add[3] - y_load[3] = 0
 R3460: x[4] - x_chg[4] + x_dis[4] - x_cut[4] + y_pos[4] - y_neg[4]
   - y_chg[4] + y_dis[4] + y_PV[4] - y_cut[4] + y_add[4] - y_load[4] = 0
 R3461: x[5] - x_chg[5] + x_dis[5] - x_cut[5] + y_pos[5] - y_neg[5]
   - y_chg[5] + y_dis[5] + y_PV[5] - y_cut[5] + y_add[5] - y_load[5] = 0
 R3462: x[6] - x_chg[6] + x_dis[6] - x_cut[6] + y_pos[6] - y_neg[6]
   - y_chg[6] + y_dis[6] + y_PV[6] - y_cut[6] + y_add[6] - y_load[6] = 0
 R3463: x[7] - x_chg[7] + x_dis[7] - x_cut[7] + y_pos[7] - y_neg[7]
   - y_chg[7] + y_dis[7] + y_PV[7] - y_cut[7] + y_add[7] - y_load[7] = 0
 R3464: x[8] - x_chg[8] + x_dis[8] - x_cut[8] + y_pos[8] - y_neg[8]
   - y_chg[8] + y_dis[8] + y_PV[8] - y_cut[8] + y_add[8] - y_load[8] = 0
 R3465: x[9] - x_chg[9] + x_dis[9] - x_cut[9] + y_pos[9] - y_neg[9]
   - y_chg[9] + y_dis[9] + y_PV[9] - y_cut[9] + y_add[9] - y_load[9] = 0
 R3466: x[10] - x_chg[10] + x_dis[10] - x_cut[10] + y_pos[10] - y_neg[10]
   - y_chg[10] + y_dis[10] + y_PV[10] - y_cut[10] + y_add[10] - y_load[10]
   = 0
 R3467: x[11] - x_chg[11] + x_dis[11] - x_cut[11] + y_pos[11] - y_neg[11]
   - y_chg[11] + y_dis[11] + y_PV[11] - y_cut[11] + y_add[11] - y_load[11]
   = 0
 R3468: x[12] - x_chg[12] + x_dis[12] - x_cut[12] + y_pos[12] - y_neg[12]
   - y_chg[12] + y_dis[12] + y_PV[12] - y_cut[12] + y_add[12] - y_load[12]
   = 0
 R3469: x[13] - x_chg[13] + x_dis[13] - x_cut[13] + y_pos[13] - y_neg[13]
   - y_chg[13] + y_dis[13] + y_PV[13] - y_cut[13] + y_add[13] - y_load[13]
   = 0
 R3470: x[14] - x_chg[14] + x_dis[14] - x_cut[14] + y_pos[14] - y_neg[14]
   - y_chg[14] + y_dis[14] + y_PV[14] - y_cut[14] + y_add[14] - y_load[14]
   = 0
 R3471: x[15] - x_chg[15] + x_dis[15] - x_cut[15] + y_pos[15] - y_neg[15]
   - y_chg[15] + y_dis[15] + y_PV[15] - y_cut[15] + y_add[15] - y_load[15]
   = 0
 R3472: x[16] - x_chg[16] + x_dis[16] - x_cut[16] + y_pos[16] - y_neg[16]
   - y_chg[16] + y_dis[16] + y_PV[16] - y_cut[16] + y_add[16] - y_load[16]
   = 0
 R3473: x[17] - x_chg[17] + x_dis[17] - x_cut[17] + y_pos[17] - y_neg[17]
   - y_chg[17] + y_dis[17] + y_PV[17] - y_cut[17] + y_add[17] - y_load[17]
   = 0
 R3474: x[18] - x_chg[18] + x_dis[18] - x_cut[18] + y_pos[18] - y_neg[18]
   - y_chg[18] + y_dis[18] + y_PV[18] - y_cut[18] + y_add[18] - y_load[18]
   = 0
 R3475: x[19] - x_chg[19] + x_dis[19] - x_cut[19] + y_pos[19] - y_neg[19]
   - y_chg[19] + y_dis[19] + y_PV[19] - y_cut[19] + y_add[19] - y_load[19]
   = 0
 R3476: x[20] - x_chg[20] + x_dis[20] - x_cut[20] + y_pos[20] - y_neg[20]
   - y_chg[20] + y_dis[20] + y_PV[20] - y_cut[20] + y_add[20] - y_load[20]
   = 0
 R3477: x[21] - x_chg[21] + x_dis[21] - x_cut[21] + y_pos[21] - y_neg[21]
   - y_chg[21] + y_dis[21] + y_PV[21] - y_cut[21] + y_add[21] - y_load[21]
   = 0
 R3478: x[22] - x_chg[22] + x_dis[22] - x_cut[22] + y_pos[22] - y_neg[22]
   - y_chg[22] + y_dis[22] + y_PV[22] - y_cut[22] + y_add[22] - y_load[22]
   = 0
 R3479: x[23] - x_chg[23] + x_dis[23] - x_cut[23] + y_pos[23] - y_neg[23]
   - y_chg[23] + y_dis[23] + y_PV[23] - y_cut[23] + y_add[23] - y_load[23]
   = 0
 R3480: x[24] - x_chg[24] + x_dis[24] - x_cut[24] + y_pos[24] - y_neg[24]
   - y_chg[24] + y_dis[24] + y_PV[24] - y_cut[24] + y_add[24] - y_load[24]
   = 0
 R3481: x[25] - x_chg[25] + x_dis[25] - x_cut[25] + y_pos[25] - y_neg[25]
   - y_chg[25] + y_dis[25] + y_PV[25] - y_cut[25] + y_add[25] - y_load[25]
   = 0
 R3482: x[26] - x_chg[26] + x_dis[26] - x_cut[26] + y_pos[26] - y_neg[26]
   - y_chg[26] + y_dis[26] + y_PV[26] - y_cut[26] + y_add[26] - y_load[26]
   = 0
 R3483: x[27] - x_chg[27] + x_dis[27] - x_cut[27] + y_pos[27] - y_neg[27]
   - y_chg[27] + y_dis[27] + y_PV[27] - y_cut[27] + y_add[27] - y_load[27]
   = 0
 R3484: x[28] - x_chg[28] + x_dis[28] - x_cut[28] + y_pos[28] - y_neg[28]
   - y_chg[28] + y_dis[28] + y_PV[28] - y_cut[28] + y_add[28] - y_load[28]
   = 0
 R3485: x[29] - x_chg[29] + x_dis[29] - x_cut[29] + y_pos[29] - y_neg[29]
   - y_chg[29] + y_dis[29] + y_PV[29] - y_cut[29] + y_add[29] - y_load[29]
   = 0
 R3486: x[30] - x_chg[30] + x_dis[30] - x_cut[30] + y_pos[30] - y_neg[30]
   - y_chg[30] + y_dis[30] + y_PV[30] - y_cut[30] + y_add[30] - y_load[30]
   = 0
 R3487: x[31] - x_chg[31] + x_dis[31] - x_cut[31] + y_pos[31] - y_neg[31]
   - y_chg[31] + y_dis[31] + y_PV[31] - y_cut[31] + y_add[31] - y_load[31]
   = 0
 R3488: x[32] - x_chg[32] + x_dis[32] - x_cut[32] + y_pos[32] - y_neg[32]
   - y_chg[32] + y_dis[32] + y_PV[32] - y_cut[32] + y_add[32] - y_load[32]
   = 0
 R3489: x[33] - x_chg[33] + x_dis[33] - x_cut[33] + y_pos[33] - y_neg[33]
   - y_chg[33] + y_dis[33] + y_PV[33] - y_cut[33] + y_add[33] - y_load[33]
   = 0
 R3490: x[34] - x_chg[34] + x_dis[34] - x_cut[34] + y_pos[34] - y_neg[34]
   - y_chg[34] + y_dis[34] + y_PV[34] - y_cut[34] + y_add[34] - y_load[34]
   = 0
 R3491: x[35] - x_chg[35] + x_dis[35] - x_cut[35] + y_pos[35] - y_neg[35]
   - y_chg[35] + y_dis[35] + y_PV[35] - y_cut[35] + y_add[35] - y_load[35]
   = 0
 R3492: x[36] - x_chg[36] + x_dis[36] - x_cut[36] + y_pos[36] - y_neg[36]
   - y_chg[36] + y_dis[36] + y_PV[36] - y_cut[36] + y_add[36] - y_load[36]
   = 0
 R3493: x[37] - x_chg[37] + x_dis[37] - x_cut[37] + y_pos[37] - y_neg[37]
   - y_chg[37] + y_dis[37] + y_PV[37] - y_cut[37] + y_add[37] - y_load[37]
   = 0
 R3494: x[38] - x_chg[38] + x_dis[38] - x_cut[38] + y_pos[38] - y_neg[38]
   - y_chg[38] + y_dis[38] + y_PV[38] - y_cut[38] + y_add[38] - y_load[38]
   = 0
 R3495: x[39] - x_chg[39] + x_dis[39] - x_cut[39] + y_pos[39] - y_neg[39]
   - y_chg[39] + y_dis[39] + y_PV[39] - y_cut[39] + y_add[39] - y_load[39]
   = 0
 R3496: x[40] - x_chg[40] + x_dis[40] - x_cut[40] + y_pos[40] - y_neg[40]
   - y_chg[40] + y_dis[40] + y_PV[40] - y_cut[40] + y_add[40] - y_load[40]
   = 0
 R3497: x[41] - x_chg[41] + x_dis[41] - x_cut[41] + y_pos[41] - y_neg[41]
   - y_chg[41] + y_dis[41] + y_PV[41] - y_cut[41] + y_add[41] - y_load[41]
   = 0
 R3498: x[42] - x_chg[42] + x_dis[42] - x_cut[42] + y_pos[42] - y_neg[42]
   - y_chg[42] + y_dis[42] + y_PV[42] - y_cut[42] + y_add[42] - y_load[42]
   = 0
 R3499: x[43] - x_chg[43] + x_dis[43] - x_cut[43] + y_pos[43] - y_neg[43]
   - y_chg[43] + y_dis[43] + y_PV[43] - y_cut[43] + y_add[43] - y_load[43]
   = 0
 R3500: x[44] - x_chg[44] + x_dis[44] - x_cut[44] + y_pos[44] - y_neg[44]
   - y_chg[44] + y_dis[44] + y_PV[44] - y_cut[44] + y_add[44] - y_load[44]
   = 0
 R3501: x[45] - x_chg[45] + x_dis[45] - x_cut[45] + y_pos[45] - y_neg[45]
   - y_chg[45] + y_dis[45] + y_PV[45] - y_cut[45] + y_add[45] - y_load[45]
   = 0
 R3502: x[46] - x_chg[46] + x_dis[46] - x_cut[46] + y_pos[46] - y_neg[46]
   - y_chg[46] + y_dis[46] + y_PV[46] - y_cut[46] + y_add[46] - y_load[46]
   = 0
 R3503: x[47] - x_chg[47] + x_dis[47] - x_cut[47] + y_pos[47] - y_neg[47]
   - y_chg[47] + y_dis[47] + y_PV[47] - y_cut[47] + y_add[47] - y_load[47]
   = 0
 R3504: x[48] - x_chg[48] + x_dis[48] - x_cut[48] + y_pos[48] - y_neg[48]
   - y_chg[48] + y_dis[48] + y_PV[48] - y_cut[48] + y_add[48] - y_load[48]
   = 0
 R3505: x[49] - x_chg[49] + x_dis[49] - x_cut[49] + y_pos[49] - y_neg[49]
   - y_chg[49] + y_dis[49] + y_PV[49] - y_cut[49] + y_add[49] - y_load[49]
   = 0
 R3506: x[50] - x_chg[50] + x_dis[50] - x_cut[50] + y_pos[50] - y_neg[50]
   - y_chg[50] + y_dis[50] + y_PV[50] - y_cut[50] + y_add[50] - y_load[50]
   = 0
 R3507: x[51] - x_chg[51] + x_dis[51] - x_cut[51] + y_pos[51] - y_neg[51]
   - y_chg[51] + y_dis[51] + y_PV[51] - y_cut[51] + y_add[51] - y_load[51]
   = 0
 R3508: x[52] - x_chg[52] + x_dis[52] - x_cut[52] + y_pos[52] - y_neg[52]
   - y_chg[52] + y_dis[52] + y_PV[52] - y_cut[52] + y_add[52] - y_load[52]
   = 0
 R3509: x[53] - x_chg[53] + x_dis[53] - x_cut[53] + y_pos[53] - y_neg[53]
   - y_chg[53] + y_dis[53] + y_PV[53] - y_cut[53] + y_add[53] - y_load[53]
   = 0
 R3510: x[54] - x_chg[54] + x_dis[54] - x_cut[54] + y_pos[54] - y_neg[54]
   - y_chg[54] + y_dis[54] + y_PV[54] - y_cut[54] + y_add[54] - y_load[54]
   = 0
 R3511: x[55] - x_chg[55] + x_dis[55] - x_cut[55] + y_pos[55] - y_neg[55]
   - y_chg[55] + y_dis[55] + y_PV[55] - y_cut[55] + y_add[55] - y_load[55]
   = 0
 R3512: x[56] - x_chg[56] + x_dis[56] - x_cut[56] + y_pos[56] - y_neg[56]
   - y_chg[56] + y_dis[56] + y_PV[56] - y_cut[56] + y_add[56] - y_load[56]
   = 0
 R3513: x[57] - x_chg[57] + x_dis[57] - x_cut[57] + y_pos[57] - y_neg[57]
   - y_chg[57] + y_dis[57] + y_PV[57] - y_cut[57] + y_add[57] - y_load[57]
   = 0
 R3514: x[58] - x_chg[58] + x_dis[58] - x_cut[58] + y_pos[58] - y_neg[58]
   - y_chg[58] + y_dis[58] + y_PV[58] - y_cut[58] + y_add[58] - y_load[58]
   = 0
 R3515: x[59] - x_chg[59] + x_dis[59] - x_cut[59] + y_pos[59] - y_neg[59]
   - y_chg[59] + y_dis[59] + y_PV[59] - y_cut[59] + y_add[59] - y_load[59]
   = 0
 R3516: x[60] - x_chg[60] + x_dis[60] - x_cut[60] + y_pos[60] - y_neg[60]
   - y_chg[60] + y_dis[60] + y_PV[60] - y_cut[60] + y_add[60] - y_load[60]
   = 0
 R3517: x[61] - x_chg[61] + x_dis[61] - x_cut[61] + y_pos[61] - y_neg[61]
   - y_chg[61] + y_dis[61] + y_PV[61] - y_cut[61] + y_add[61] - y_load[61]
   = 0
 R3518: x[62] - x_chg[62] + x_dis[62] - x_cut[62] + y_pos[62] - y_neg[62]
   - y_chg[62] + y_dis[62] + y_PV[62] - y_cut[62] + y_add[62] - y_load[62]
   = 0
 R3519: x[63] - x_chg[63] + x_dis[63] - x_cut[63] + y_pos[63] - y_neg[63]
   - y_chg[63] + y_dis[63] + y_PV[63] - y_cut[63] + y_add[63] - y_load[63]
   = 0
 R3520: x[64] - x_chg[64] + x_dis[64] - x_cut[64] + y_pos[64] - y_neg[64]
   - y_chg[64] + y_dis[64] + y_PV[64] - y_cut[64] + y_add[64] - y_load[64]
   = 0
 R3521: x[65] - x_chg[65] + x_dis[65] - x_cut[65] + y_pos[65] - y_neg[65]
   - y_chg[65] + y_dis[65] + y_PV[65] - y_cut[65] + y_add[65] - y_load[65]
   = 0
 R3522: x[66] - x_chg[66] + x_dis[66] - x_cut[66] + y_pos[66] - y_neg[66]
   - y_chg[66] + y_dis[66] + y_PV[66] - y_cut[66] + y_add[66] - y_load[66]
   = 0
 R3523: x[67] - x_chg[67] + x_dis[67] - x_cut[67] + y_pos[67] - y_neg[67]
   - y_chg[67] + y_dis[67] + y_PV[67] - y_cut[67] + y_add[67] - y_load[67]
   = 0
 R3524: x[68] - x_chg[68] + x_dis[68] - x_cut[68] + y_pos[68] - y_neg[68]
   - y_chg[68] + y_dis[68] + y_PV[68] - y_cut[68] + y_add[68] - y_load[68]
   = 0
 R3525: x[69] - x_chg[69] + x_dis[69] - x_cut[69] + y_pos[69] - y_neg[69]
   - y_chg[69] + y_dis[69] + y_PV[69] - y_cut[69] + y_add[69] - y_load[69]
   = 0
 R3526: x[70] - x_chg[70] + x_dis[70] - x_cut[70] + y_pos[70] - y_neg[70]
   - y_chg[70] + y_dis[70] + y_PV[70] - y_cut[70] + y_add[70] - y_load[70]
   = 0
 R3527: x[71] - x_chg[71] + x_dis[71] - x_cut[71] + y_pos[71] - y_neg[71]
   - y_chg[71] + y_dis[71] + y_PV[71] - y_cut[71] + y_add[71] - y_load[71]
   = 0
 R3528: x[72] - x_chg[72] + x_dis[72] - x_cut[72] + y_pos[72] - y_neg[72]
   - y_chg[72] + y_dis[72] + y_PV[72] - y_cut[72] + y_add[72] - y_load[72]
   = 0
 R3529: x[73] - x_chg[73] + x_dis[73] - x_cut[73] + y_pos[73] - y_neg[73]
   - y_chg[73] + y_dis[73] + y_PV[73] - y_cut[73] + y_add[73] - y_load[73]
   = 0
 R3530: x[74] - x_chg[74] + x_dis[74] - x_cut[74] + y_pos[74] - y_neg[74]
   - y_chg[74] + y_dis[74] + y_PV[74] - y_cut[74] + y_add[74] - y_load[74]
   = 0
 R3531: x[75] - x_chg[75] + x_dis[75] - x_cut[75] + y_pos[75] - y_neg[75]
   - y_chg[75] + y_dis[75] + y_PV[75] - y_cut[75] + y_add[75] - y_load[75]
   = 0
 R3532: x[76] - x_chg[76] + x_dis[76] - x_cut[76] + y_pos[76] - y_neg[76]
   - y_chg[76] + y_dis[76] + y_PV[76] - y_cut[76] + y_add[76] - y_load[76]
   = 0
 R3533: x[77] - x_chg[77] + x_dis[77] - x_cut[77] + y_pos[77] - y_neg[77]
   - y_chg[77] + y_dis[77] + y_PV[77] - y_cut[77] + y_add[77] - y_load[77]
   = 0
 R3534: x[78] - x_chg[78] + x_dis[78] - x_cut[78] + y_pos[78] - y_neg[78]
   - y_chg[78] + y_dis[78] + y_PV[78] - y_cut[78] + y_add[78] - y_load[78]
   = 0
 R3535: x[79] - x_chg[79] + x_dis[79] - x_cut[79] + y_pos[79] - y_neg[79]
   - y_chg[79] + y_dis[79] + y_PV[79] - y_cut[79] + y_add[79] - y_load[79]
   = 0
 R3536: x[80] - x_chg[80] + x_dis[80] - x_cut[80] + y_pos[80] - y_neg[80]
   - y_chg[80] + y_dis[80] + y_PV[80] - y_cut[80] + y_add[80] - y_load[80]
   = 0
 R3537: x[81] - x_chg[81] + x_dis[81] - x_cut[81] + y_pos[81] - y_neg[81]
   - y_chg[81] + y_dis[81] + y_PV[81] - y_cut[81] + y_add[81] - y_load[81]
   = 0
 R3538: x[82] - x_chg[82] + x_dis[82] - x_cut[82] + y_pos[82] - y_neg[82]
   - y_chg[82] + y_dis[82] + y_PV[82] - y_cut[82] + y_add[82] - y_load[82]
   = 0
 R3539: x[83] - x_chg[83] + x_dis[83] - x_cut[83] + y_pos[83] - y_neg[83]
   - y_chg[83] + y_dis[83] + y_PV[83] - y_cut[83] + y_add[83] - y_load[83]
   = 0
 R3540: x[84] - x_chg[84] + x_dis[84] - x_cut[84] + y_pos[84] - y_neg[84]
   - y_chg[84] + y_dis[84] + y_PV[84] - y_cut[84] + y_add[84] - y_load[84]
   = 0
 R3541: x[85] - x_chg[85] + x_dis[85] - x_cut[85] + y_pos[85] - y_neg[85]
   - y_chg[85] + y_dis[85] + y_PV[85] - y_cut[85] + y_add[85] - y_load[85]
   = 0
 R3542: x[86] - x_chg[86] + x_dis[86] - x_cut[86] + y_pos[86] - y_neg[86]
   - y_chg[86] + y_dis[86] + y_PV[86] - y_cut[86] + y_add[86] - y_load[86]
   = 0
 R3543: x[87] - x_chg[87] + x_dis[87] - x_cut[87] + y_pos[87] - y_neg[87]
   - y_chg[87] + y_dis[87] + y_PV[87] - y_cut[87] + y_add[87] - y_load[87]
   = 0
 R3544: x[88] - x_chg[88] + x_dis[88] - x_cut[88] + y_pos[88] - y_neg[88]
   - y_chg[88] + y_dis[88] + y_PV[88] - y_cut[88] + y_add[88] - y_load[88]
   = 0
 R3545: x[89] - x_chg[89] + x_dis[89] - x_cut[89] + y_pos[89] - y_neg[89]
   - y_chg[89] + y_dis[89] + y_PV[89] - y_cut[89] + y_add[89] - y_load[89]
   = 0
 R3546: x[90] - x_chg[90] + x_dis[90] - x_cut[90] + y_pos[90] - y_neg[90]
   - y_chg[90] + y_dis[90] + y_PV[90] - y_cut[90] + y_add[90] - y_load[90]
   = 0
 R3547: x[91] - x_chg[91] + x_dis[91] - x_cut[91] + y_pos[91] - y_neg[91]
   - y_chg[91] + y_dis[91] + y_PV[91] - y_cut[91] + y_add[91] - y_load[91]
   = 0
 R3548: x[92] - x_chg[92] + x_dis[92] - x_cut[92] + y_pos[92] - y_neg[92]
   - y_chg[92] + y_dis[92] + y_PV[92] - y_cut[92] + y_add[92] - y_load[92]
   = 0
 R3549: x[93] - x_chg[93] + x_dis[93] - x_cut[93] + y_pos[93] - y_neg[93]
   - y_chg[93] + y_dis[93] + y_PV[93] - y_cut[93] + y_add[93] - y_load[93]
   = 0
 R3550: x[94] - x_chg[94] + x_dis[94] - x_cut[94] + y_pos[94] - y_neg[94]
   - y_chg[94] + y_dis[94] + y_PV[94] - y_cut[94] + y_add[94] - y_load[94]
   = 0
 R3551: x[95] - x_chg[95] + x_dis[95] - x_cut[95] + y_pos[95] - y_neg[95]
   - y_chg[95] + y_dis[95] + y_PV[95] - y_cut[95] + y_add[95] - y_load[95]
   = 0
Bounds
Binaries
 x_b[0] x_b[1] x_b[2] x_b[3] x_b[4] x_b[5] x_b[6] x_b[7] x_b[8] x_b[9]
 x_b[10] x_b[11] x_b[12] x_b[13] x_b[14] x_b[15] x_b[16] x_b[17] x_b[18]
 x_b[19] x_b[20] x_b[21] x_b[22] x_b[23] x_b[24] x_b[25] x_b[26] x_b[27]
 x_b[28] x_b[29] x_b[30] x_b[31] x_b[32] x_b[33] x_b[34] x_b[35] x_b[36]
 x_b[37] x_b[38] x_b[39] x_b[40] x_b[41] x_b[42] x_b[43] x_b[44] x_b[45]
 x_b[46] x_b[47] x_b[48] x_b[49] x_b[50] x_b[51] x_b[52] x_b[53] x_b[54]
 x_b[55] x_b[56] x_b[57] x_b[58] x_b[59] x_b[60] x_b[61] x_b[62] x_b[63]
 x_b[64] x_b[65] x_b[66] x_b[67] x_b[68] x_b[69] x_b[70] x_b[71] x_b[72]
 x_b[73] x_b[74] x_b[75] x_b[76] x_b[77] x_b[78] x_b[79] x_b[80] x_b[81]
 x_b[82] x_b[83] x_b[84] x_b[85] x_b[86] x_b[87] x_b[88] x_b[89] x_b[90]
 x_b[91] x_b[92] x_b[93] x_b[94] x_b[95] y_b[0] y_b[1] y_b[2] y_b[3] y_b[4]
 y_b[5] y_b[6] y_b[7] y_b[8] y_b[9] y_b[10] y_b[11] y_b[12] y_b[13] y_b[14]
 y_b[15] y_b[16] y_b[17] y_b[18] y_b[19] y_b[20] y_b[21] y_b[22] y_b[23]
 y_b[24] y_b[25] y_b[26] y_b[27] y_b[28] y_b[29] y_b[30] y_b[31] y_b[32]
 y_b[33] y_b[34] y_b[35] y_b[36] y_b[37] y_b[38] y_b[39] y_b[40] y_b[41]
 y_b[42] y_b[43] y_b[44] y_b[45] y_b[46] y_b[47] y_b[48] y_b[49] y_b[50]
 y_b[51] y_b[52] y_b[53] y_b[54] y_b[55] y_b[56] y_b[57] y_b[58] y_b[59]
 y_b[60] y_b[61] y_b[62] y_b[63] y_b[64] y_b[65] y_b[66] y_b[67] y_b[68]
 y_b[69] y_b[70] y_b[71] y_b[72] y_b[73] y_b[74] y_b[75] y_b[76] y_b[77]
 y_b[78] y_b[79] y_b[80] y_b[81] y_b[82] y_b[83] y_b[84] y_b[85] y_b[86]
 y_b[87] y_b[88] y_b[89] y_b[90] y_b[91] y_b[92] y_b[93] y_b[94] y_b[95]
General Constraints
 GC0: x_cost_fuel[0] = PWL ( x[0] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC1: x_cost_cut[0] = PWL ( x_cut[0] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC2: x_cost_fuel[1] = PWL ( x[1] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC3: x_cost_cut[1] = PWL ( x_cut[1] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC4: x_cost_fuel[2] = PWL ( x[2] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC5: x_cost_cut[2] = PWL ( x_cut[2] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC6: x_cost_fuel[3] = PWL ( x[3] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC7: x_cost_cut[3] = PWL ( x_cut[3] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC8: x_cost_fuel[4] = PWL ( x[4] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC9: x_cost_cut[4] = PWL ( x_cut[4] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC10: x_cost_fuel[5] = PWL ( x[5] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC11: x_cost_cut[5] = PWL ( x_cut[5] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC12: x_cost_fuel[6] = PWL ( x[6] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC13: x_cost_cut[6] = PWL ( x_cut[6] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC14: x_cost_fuel[7] = PWL ( x[7] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC15: x_cost_cut[7] = PWL ( x_cut[7] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC16: x_cost_fuel[8] = PWL ( x[8] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC17: x_cost_cut[8] = PWL ( x_cut[8] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC18: x_cost_fuel[9] = PWL ( x[9] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC19: x_cost_cut[9] = PWL ( x_cut[9] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC20: x_cost_fuel[10] = PWL ( x[10] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC21: x_cost_cut[10] = PWL ( x_cut[10] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC22: x_cost_fuel[11] = PWL ( x[11] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC23: x_cost_cut[11] = PWL ( x_cut[11] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC24: x_cost_fuel[12] = PWL ( x[12] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC25: x_cost_cut[12] = PWL ( x_cut[12] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC26: x_cost_fuel[13] = PWL ( x[13] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC27: x_cost_cut[13] = PWL ( x_cut[13] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC28: x_cost_fuel[14] = PWL ( x[14] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC29: x_cost_cut[14] = PWL ( x_cut[14] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC30: x_cost_fuel[15] = PWL ( x[15] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC31: x_cost_cut[15] = PWL ( x_cut[15] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC32: x_cost_fuel[16] = PWL ( x[16] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC33: x_cost_cut[16] = PWL ( x_cut[16] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC34: x_cost_fuel[17] = PWL ( x[17] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC35: x_cost_cut[17] = PWL ( x_cut[17] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC36: x_cost_fuel[18] = PWL ( x[18] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC37: x_cost_cut[18] = PWL ( x_cut[18] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC38: x_cost_fuel[19] = PWL ( x[19] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC39: x_cost_cut[19] = PWL ( x_cut[19] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC40: x_cost_fuel[20] = PWL ( x[20] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC41: x_cost_cut[20] = PWL ( x_cut[20] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC42: x_cost_fuel[21] = PWL ( x[21] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC43: x_cost_cut[21] = PWL ( x_cut[21] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC44: x_cost_fuel[22] = PWL ( x[22] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC45: x_cost_cut[22] = PWL ( x_cut[22] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC46: x_cost_fuel[23] = PWL ( x[23] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC47: x_cost_cut[23] = PWL ( x_cut[23] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC48: x_cost_fuel[24] = PWL ( x[24] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC49: x_cost_cut[24] = PWL ( x_cut[24] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC50: x_cost_fuel[25] = PWL ( x[25] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC51: x_cost_cut[25] = PWL ( x_cut[25] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC52: x_cost_fuel[26] = PWL ( x[26] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC53: x_cost_cut[26] = PWL ( x_cut[26] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC54: x_cost_fuel[27] = PWL ( x[27] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC55: x_cost_cut[27] = PWL ( x_cut[27] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC56: x_cost_fuel[28] = PWL ( x[28] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC57: x_cost_cut[28] = PWL ( x_cut[28] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC58: x_cost_fuel[29] = PWL ( x[29] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC59: x_cost_cut[29] = PWL ( x_cut[29] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC60: x_cost_fuel[30] = PWL ( x[30] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC61: x_cost_cut[30] = PWL ( x_cut[30] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC62: x_cost_fuel[31] = PWL ( x[31] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC63: x_cost_cut[31] = PWL ( x_cut[31] ) : (0, 0) 
  (0.00253118027355328, 6.40687e-09) (0.00506236054710657, 2.56275e-08) 
  (0.00759354082065985, 5.76619e-08) (0.0101247210942131, 1.0251e-07) 
  (0.0126559013677664, 1.60172e-07) (0.0151870816413197, 2.30647e-07) 
  (0.017718261914873, 3.13937e-07) (0.0202494421884263, 4.1004e-07) 
  (0.0227806224619796, 5.18957e-07) (0.0253118027355328, 6.40687e-07)
 GC64: x_cost_fuel[32] = PWL ( x[32] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC65: x_cost_cut[32] = PWL ( x_cut[32] ) : (0, 0) 
  (0.356637887912738, 1.2719058309485867e-04) 
  (0.713275775825476, 5.0876233237943469e-04) 
  (1.069913663738214, 0.00114471524785373) 
  (1.426551551650952, 0.00203504932951774) 
  (1.78318943956369, 0.00317976457737147) 
  (2.139827327476428, 0.00457886099141491) 
  (2.496465215389166, 0.00623233857164807) 
  (2.853103103301904, 0.00814019731807096) 
  (3.209740991214642, 0.0103024372306836) 
  (3.56637887912738, 0.0127190583094859)
 GC66: x_cost_fuel[33] = PWL ( x[33] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC67: x_cost_cut[33] = PWL ( x_cut[33] ) : (0, 0) 
  (0.6066765780669403, 3.6805647037501234e-04) 
  (1.213353156133881, 0.00147222588150005) 
  (1.820029734200821, 0.00331250823337511) 
  (2.426706312267761, 0.0058889035260002) 
  (3.033382890334702, 0.00920141175937531) 
  (3.640059468401642, 0.0132500329335004) 
  (4.246736046468582, 0.0180347670483756) 
  (4.853412624535522, 0.0235556141040008) 
  (5.460089202602463, 0.029812574100376) 
  (6.066765780669403, 0.0368056470375012)
 GC68: x_cost_fuel[34] = PWL ( x[34] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC69: x_cost_cut[34] = PWL ( x_cut[34] ) : (0, 0) 
  (1.102108279071951, 0.00121464265879894) 
  (2.204216558143901, 0.00485857063519575) 
  (3.306324837215853, 0.0109317839291904) 
  (4.408433116287803, 0.019434282540783) 
  (5.510541395359754, 0.0303660664699734) 
  (6.612649674431705, 0.0437271357167617) 
  (7.714757953503655, 0.0595174902811479) 
  (8.816866232575606, 0.077737130163132) 
  (9.918974511647557, 0.0983860553627139) 
  (11.02108279071951, 0.1214642658798937)
 GC70: x_cost_fuel[35] = PWL ( x[35] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC71: x_cost_cut[35] = PWL ( x_cut[35] ) : (0, 0) 
  (4.014181803835542, 0.0161136555542444) 
  (8.028363607671084, 0.0644546222169775) 
  (12.04254541150663, 0.1450228999881993) 
  (16.05672721534217, 0.2578184888679098) 
  (20.07090901917771, 0.4028413888561091) 
  (24.08509082301325, 0.5800915999527972) 
  (28.09927262684879, 0.7895691221579737) 
  (32.11345443068434, 1.031273955471639) 
  (36.12763623451988, 1.305206099893794) 
  (40.14181803835542, 1.611365555424436)
 GC72: x_cost_fuel[36] = PWL ( x[36] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC73: x_cost_cut[36] = PWL ( x_cut[36] ) : (0, 0) 
  (5.864215775270846, 0.0343890266589355) 
  (11.72843155054169, 0.1375561066357418) 
  (17.59264732581254, 0.3095012399304191) 
  (23.45686310108339, 0.5502244265429673) 
  (29.32107887635423, 0.8597256664733863) 
  (35.18529465162508, 1.238004959721676) 
  (41.04951042689593, 1.685062306287837) 
  (46.91372620216677, 2.200897706171869) 
  (52.77794197743761, 2.785511159373771) 
  (58.64215775270846, 3.438902665893545)
 GC74: x_cost_fuel[37] = PWL ( x[37] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC75: x_cost_cut[37] = PWL ( x_cut[37] ) : (0, 0) 
  (7.118195246506995, 0.0506687035673948) 
  (14.23639049301399, 0.2026748142695791) 
  (21.35458573952099, 0.456018332106553) 
  (28.47278098602798, 0.8106992570783166) 
  (35.59097623253498, 1.26671758918487) 
  (42.70917147904197, 1.824073328426212) 
  (49.82736672554897, 2.482766474802345) 
  (56.94556197205596, 3.242797028313266) 
  (64.06375721856296, 4.104164988958978) 
  (71.18195246506995, 5.066870356739479)
 GC76: x_cost_fuel[38] = PWL ( x[38] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC77: x_cost_cut[38] = PWL ( x_cut[38] ) : (0, 0) 
  (7.863593092742535, 0.0618360963282281) 
  (15.72718618548507, 0.2473443853129125) 
  (23.5907792782276, 0.5565248669540529) 
  (31.45437237097014, 0.9893775412516499) 
  (39.31796546371267, 1.545902408205703) 
  (47.18155855645521, 2.226099467816212) 
  (55.04515164919775, 3.029968720083178) 
  (62.90874474194028, 3.9575101650066) 
  (70.77233783468282, 5.008723802586476) 
  (78.63593092742535, 6.183609632822811)
 GC78: x_cost_fuel[39] = PWL ( x[39] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC79: x_cost_cut[39] = PWL ( x_cut[39] ) : (0, 0) 
  (10.39598003496597, 0.108076400887411) 
  (20.79196006993193, 0.4323056035496439) 
  (31.1879401048979, 0.9726876079866986) 
  (41.58392013986386, 1.729222414198576) 
  (51.97990017482984, 2.701910022185275) 
  (62.3758802097958, 3.890750431946794) 
  (72.77186024476177, 5.295743643483139) 
  (83.16784027972773, 6.916889656794303) 
  (93.56382031469369, 8.754188471880287) 
  (103.9598003496597, 10.8076400887411)
 GC80: x_cost_fuel[40] = PWL ( x[40] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC81: x_cost_cut[40] = PWL ( x_cut[40] ) : (0, 0) 
  (12.0796496603743, 0.1459179359173811) 
  (24.15929932074861, 0.5836717436695242) 
  (36.23894898112292, 1.31326142325643) 
  (48.31859864149722, 2.334686974678097) 
  (60.39824830187153, 3.647948397934527) 
  (72.47789796224583, 5.253045693025719) 
  (84.55754762262013, 7.149978859951672) 
  (96.63719728299444, 9.338747898712388) 
  (108.7168469433688, 11.81935280930787) 
  (120.7964966037431, 14.59179359173811)
 GC82: x_cost_fuel[41] = PWL ( x[41] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC83: x_cost_cut[41] = PWL ( x_cut[41] ) : (0, 0) 
  (13.08044015164348, 0.1710979145607269) 
  (26.16088030328696, 0.6843916582429077) 
  (39.24132045493044, 1.539881231046542) 
  (52.32176060657392, 2.737566632971631) 
  (65.4022007582174, 4.277447864018173) 
  (78.48264090986088, 6.159524924186168) 
  (91.56308106150436, 8.383797813475619) 
  (104.6435212131478, 10.95026653188652) 
  (117.7239613647913, 13.85893107941888) 
  (130.8044015164348, 17.10979145607269)
 GC84: x_cost_fuel[42] = PWL ( x[42] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC85: x_cost_cut[42] = PWL ( x_cut[42] ) : (0, 0) 
  (13.52361679921553, 0.1828882113320244) 
  (27.04723359843105, 0.7315528453280976) 
  (40.57085039764657, 1.645993901988219) 
  (54.09446719686211, 2.92621138131239) 
  (67.61808399607763, 4.57220528330061) 
  (81.14170079529315, 6.583975607952876) 
  (94.66531759450868, 8.961522355269194) 
  (108.1889343937242, 11.70484552524956) 
  (121.7125511929397, 14.81394511789398) 
  (135.2361679921553, 18.28882113320244)
 GC86: x_cost_fuel[43] = PWL ( x[43] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC87: x_cost_cut[43] = PWL ( x_cut[43] ) : (0, 0) 
  (15.75312789367545, 0.2481610384344954) 
  (31.50625578735089, 0.9926441537379815) 
  (47.25938368102634, 2.233449345910458) 
  (63.01251157470178, 3.970576614951926) 
  (78.76563946837723, 6.204025960862386) 
  (94.51876736205267, 8.933797383641833) 
  (110.2718952557281, 12.15989088329028) 
  (126.0250231494036, 15.8823064598077) 
  (141.778151043079, 20.10104411319413) 
  (157.5312789367545, 24.81610384344954)
 GC88: x_cost_fuel[44] = PWL ( x[44] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC89: x_cost_cut[44] = PWL ( x_cut[44] ) : (0, 0) 
  (17.37538437201719, 0.3019039820753391) 
  (34.75076874403437, 1.207615928301357) 
  (52.12615311605157, 2.717135838678053) 
  (69.50153748806875, 4.830463713205426) 
  (86.87692186008594, 7.547599551883478) 
  (104.2523062321031, 10.86854335471221) 
  (121.6276906041203, 14.79329512169162) 
  (139.0030749761375, 19.3218548528217) 
  (156.3784593481547, 24.45422254810247) 
  (173.7538437201719, 30.19039820753391)
 GC90: x_cost_fuel[45] = PWL ( x[45] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC91: x_cost_cut[45] = PWL ( x_cut[45] ) : (0, 0) 
  (17.96553866724427, 0.3227605796042489) 
  (35.93107733448853, 1.291042318416996) 
  (53.89661600173281, 2.904845216438241) 
  (71.86215466897707, 5.164169273667983) 
  (89.82769333622134, 8.069014490106225) 
  (107.7932320034656, 11.61938086575296) 
  (125.7587706707099, 15.8152684006082) 
  (143.7243093379541, 20.65667709467193) 
  (161.6898480051984, 26.14360694794417) 
  (179.6553866724427, 32.2760579604249)
 GC92: x_cost_fuel[46] = PWL ( x[46] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC93: x_cost_cut[46] = PWL ( x_cut[46] ) : (0, 0) 
  (17.57944314299832, 0.3090368212179107) 
  (35.15888628599664, 1.236147284871643) 
  (52.73832942899496, 2.781331390961196) 
  (70.31777257199329, 4.944589139486571) 
  (87.8972157149916, 7.725920530447767) 
  (105.4766588579899, 11.12532556384478) 
  (123.0561020009882, 15.14280423967762) 
  (140.6355451439866, 19.77835655794629) 
  (158.2149882869849, 25.03198251865076) 
  (175.7944314299832, 30.90368212179107)
 GC94: x_cost_fuel[47] = PWL ( x[47] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC95: x_cost_cut[47] = PWL ( x_cut[47] ) : (0, 0) 
  (20.07069986029999, 0.402832992882246) 
  (40.14139972059998, 1.611331971528984) 
  (60.21209958089996, 3.625496935940213) 
  (80.28279944119996, 6.445327886115936) 
  (100.3534993014999, 10.07082482205615) 
  (120.4241991617999, 14.50198774376085) 
  (140.4948990220999, 19.73881665123005) 
  (160.5655988823999, 25.78131154446374) 
  (180.6362987426999, 32.62947242346193) 
  (200.7069986029999, 40.28329928822459)
 GC96: x_cost_fuel[48] = PWL ( x[48] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC97: x_cost_cut[48] = PWL ( x_cut[48] ) : (0, 0) 
  (21.43541259015806, 0.4594769129103066) 
  (42.87082518031612, 1.837907651641226) 
  (64.30623777047417, 4.135292216192759) 
  (85.74165036063224, 7.351630606564906) 
  (107.1770629507903, 11.48692282275766) 
  (128.6124755409483, 16.54116886477103) 
  (150.0478881311064, 22.51436873260502) 
  (171.4833007212645, 29.40652242625962) 
  (192.9187133114225, 37.21762994573483) 
  (214.3541259015806, 45.94769129103065)
 GC98: x_cost_fuel[49] = PWL ( x[49] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC99: x_cost_cut[49] = PWL ( x_cut[49] ) : (0, 0) 
  (21.43913137253857, 0.4596363540089677) 
  (42.87826274507714, 1.838545416035871) 
  (64.31739411761571, 4.136727186080708) 
  (85.75652549015429, 7.354181664143483) 
  (107.1956568626929, 11.49090885022419) 
  (128.6347882352314, 16.54690874432283) 
  (150.07391960777, 22.52218134643941) 
  (171.5130509803086, 29.41672665657393) 
  (192.9521823528472, 37.23054467472638) 
  (214.3913137253857, 45.96363540089676)
 GC100: x_cost_fuel[50] = PWL ( x[50] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC101: x_cost_cut[50] = PWL ( x_cut[50] ) : (0, 0) 
  (20.76329982380296, 0.431114619573136) 
  (41.52659964760592, 1.724458478292544) 
  (62.28989947140887, 3.880031576158224) 
  (83.05319929521184, 6.897833913170176) 
  (103.8164991190148, 10.7778654893284) 
  (124.5797989428177, 15.52012630463289) 
  (145.3430987666207, 21.12461635908366) 
  (166.1063985904237, 27.5913356526807) 
  (186.8696984142266, 34.92028418542402) 
  (207.6329982380296, 43.11146195731359)
 GC102: x_cost_fuel[51] = PWL ( x[51] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC103: x_cost_cut[51] = PWL ( x_cut[51] ) : (0, 0) 
  (22.29774754564774, 0.4971895456094397) 
  (44.59549509129548, 1.988758182437759) 
  (66.89324263694321, 4.474705910484957) 
  (89.19099018259095, 7.955032729751036) 
  (111.4887377282387, 12.429738640236) 
  (133.7864852738864, 17.89882364193983) 
  (156.0842328195342, 24.36228773486255) 
  (178.3819803651819, 31.82013091900414) 
  (200.6797279108296, 40.27235319436461) 
  (222.9774754564774, 49.71895456094398)
 GC104: x_cost_fuel[52] = PWL ( x[52] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC105: x_cost_cut[52] = PWL ( x_cut[52] ) : (0, 0) 
  (22.90993423760023, 0.5248650867711671) 
  (45.81986847520045, 2.099460347084668) 
  (68.72980271280068, 4.723785780940503) 
  (91.6397369504009, 8.397841388338673) 
  (114.5496711880011, 13.12162716927918) 
  (137.4596054256014, 18.89514312376201) 
  (160.3695396632016, 25.71838925178718) 
  (183.2794739008018, 33.59136555335469) 
  (206.189408138402, 42.51407202846453) 
  (229.0993423760023, 52.48650867711672)
 GC106: x_cost_fuel[53] = PWL ( x[53] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC107: x_cost_cut[53] = PWL ( x_cut[53] ) : (0, 0) 
  (22.64956731384092, 0.5130028995042112) 
  (45.29913462768184, 2.052011598016845) 
  (67.94870194152277, 4.6170260955379) 
  (90.59826925536369, 8.208046392067379) 
  (113.2478365692046, 12.82507248760528) 
  (135.8974038830455, 18.4681043821516) 
  (158.5469711968864, 25.13714207570634) 
  (181.1965385107274, 32.83218556826952) 
  (203.8461058245683, 41.5532348598411) 
  (226.4956731384092, 51.3002899504211)
 GC108: x_cost_fuel[54] = PWL ( x[54] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC109: x_cost_cut[54] = PWL ( x_cut[54] ) : (0, 0) 
  (21.52602660261785, 0.4633698212966116) 
  (43.05205320523571, 1.853479285186446) 
  (64.57807980785357, 4.170328391669505) 
  (86.10410641047142, 7.413917140745785) 
  (107.6301330130893, 11.58424553241529) 
  (129.1561596157071, 16.68131356667802) 
  (150.682186218325, 22.70512124353396) 
  (172.2082128209428, 29.65566856298314) 
  (193.7342394235607, 37.53295552502555) 
  (215.2602660261786, 46.33698212966117)
 GC110: x_cost_fuel[55] = PWL ( x[55] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC111: x_cost_cut[55] = PWL ( x_cut[55] ) : (0, 0) 
  (22.22076689793372, 0.4937624815323069) 
  (44.44153379586744, 1.975049926129228) 
  (66.66230069380114, 4.443862333790761) 
  (88.88306759173487, 7.90019970451691) 
  (111.1038344896686, 12.34406203830767) 
  (133.3246013876023, 17.77544933516304) 
  (155.545368285536, 24.19436159508304) 
  (177.7661351834697, 31.60079881806764) 
  (199.9869020814035, 39.99476100411685) 
  (222.2076689793372, 49.37624815323068)
 GC112: x_cost_fuel[56] = PWL ( x[56] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC113: x_cost_cut[56] = PWL ( x_cut[56] ) : (0, 0) 
  (21.41577624899606, 0.4586354723470638) 
  (42.83155249799212, 1.834541889388255) 
  (64.24732874698819, 4.127719251123574) 
  (85.66310499598424, 7.33816755755302) 
  (107.0788812449803, 11.4658868086766) 
  (128.4946574939764, 16.5108770044943) 
  (149.9104337429724, 22.47313814500613) 
  (171.3262099919685, 29.35267023021208) 
  (192.7419862409645, 37.14947326011216) 
  (214.1577624899606, 45.86354723470639)
 GC114: x_cost_fuel[57] = PWL ( x[57] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC115: x_cost_cut[57] = PWL ( x_cut[57] ) : (0, 0) 
  (19.65971352475247, 0.386504335875335) 
  (39.31942704950493, 1.54601734350134) 
  (58.97914057425739, 3.478539022878015) 
  (78.63885409900986, 6.18406937400536) 
  (98.29856762376232, 9.662608396883375) 
  (117.9582811485148, 13.91415609151206) 
  (137.6179946732672, 18.93871245789141) 
  (157.2777081980197, 24.73627749602144) 
  (176.9374217227722, 31.30685120590213) 
  (196.5971352475246, 38.6504335875335)
 GC116: x_cost_fuel[58] = PWL ( x[58] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC117: x_cost_cut[58] = PWL ( x_cut[58] ) : (0, 0) 
  (16.93389034902879, 0.2867566423529305) 
  (33.86778069805759, 1.147026569411722) 
  (50.80167104708637, 2.580809781176374) 
  (67.73556139611517, 4.588106277646887) 
  (84.66945174514396, 7.168916058823262) 
  (101.6033420941727, 10.3232391247055) 
  (118.5372324432015, 14.05107547529359) 
  (135.4711227922303, 18.35242511058755) 
  (152.4050131412591, 23.22728803058737) 
  (169.3389034902879, 28.67566423529305)
 GC118: x_cost_fuel[59] = PWL ( x[59] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC119: x_cost_cut[59] = PWL ( x_cut[59] ) : (0, 0) 
  (16.49690439176621, 0.2721478545110754) 
  (32.99380878353243, 1.088591418044302) 
  (49.49071317529865, 2.449330690599679) 
  (65.98761756706486, 4.354365672177207) 
  (82.48452195883108, 6.803696362776887) 
  (98.9814263505973, 9.797322762398718) 
  (115.4783307423635, 13.3352448710427) 
  (131.9752351341297, 17.41746268870883) 
  (148.472139525896, 22.04397621539712) 
  (164.9690439176622, 27.21478545110755)
 GC120: x_cost_fuel[60] = PWL ( x[60] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC121: x_cost_cut[60] = PWL ( x_cut[60] ) : (0, 0) 
  (12.00895113555729, 0.1442149073762028) 
  (24.01790227111459, 0.5768596295048114) 
  (36.02685340667189, 1.297934166385826) 
  (48.03580454222918, 2.307438518019246) 
  (60.04475567778648, 3.605372684405071) 
  (72.05370681334378, 5.191736665543304) 
  (84.06265794890106, 7.066530461433939) 
  (96.07160908445836, 9.229754072076982) 
  (108.0805602200157, 11.68140749747243) 
  (120.089511355573, 14.42149073762029)
 GC122: x_cost_fuel[61] = PWL ( x[61] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC123: x_cost_cut[61] = PWL ( x_cut[61] ) : (0, 0) 
  (13.58592201117795, 0.1845772768938096) 
  (27.17184402235591, 0.7383091075752384) 
  (40.75776603353385, 1.661195492044286) 
  (54.34368804471181, 2.953236430300954) 
  (67.92961005588977, 4.61443192234524) 
  (81.51553206706771, 6.644781968177145) 
  (95.10145407824567, 9.04428656779667) 
  (108.6873760894236, 11.81294572120381) 
  (122.2732981006016, 14.95075942839858) 
  (135.8592201117795, 18.45772768938096)
 GC124: x_cost_fuel[62] = PWL ( x[62] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC125: x_cost_cut[62] = PWL ( x_cut[62] ) : (0, 0) 
  (10.79922319062659, 0.1166232215209671) 
  (21.59844638125318, 0.4664928860838684) 
  (32.39766957187977, 1.049608993688704) 
  (43.19689276250635, 1.865971544335474) 
  (53.99611595313295, 2.915580538024179) 
  (64.79533914375953, 4.198435974754817) 
  (75.59456233438611, 5.714537854527387) 
  (86.3937855250127, 7.463886177341895) 
  (97.19300871563931, 9.446480943198338) 
  (107.9922319062659, 11.66232215209672)
 GC126: x_cost_fuel[63] = PWL ( x[63] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC127: x_cost_cut[63] = PWL ( x_cut[63] ) : (0, 0) 
  (9.749370147574698, 0.0950502182744207) 
  (19.4987402951494, 0.3802008730976827) 
  (29.24811044272409, 0.8554519644697862) 
  (38.99748059029879, 1.520803492390731) 
  (48.74685073787349, 2.376255456860517) 
  (58.49622088544819, 3.421807857879145) 
  (68.24559103302289, 4.657460695446614) 
  (77.99496118059758, 6.083213969562924) 
  (87.74433132817229, 7.699067680228078) 
  (97.49370147574699, 9.50502182744207)
 GC128: x_cost_fuel[64] = PWL ( x[64] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC129: x_cost_cut[64] = PWL ( x_cut[64] ) : (0, 0) 
  (8.266484505094192, 0.0683347660729624) 
  (16.53296901018838, 0.2733390642918495) 
  (24.79945351528258, 0.6150128946566613) 
  (33.06593802037677, 1.093356257167398) 
  (41.33242252547096, 1.708369151824059) 
  (49.59890703056515, 2.460051578626645) 
  (57.86539153565934, 3.348403537575156) 
  (66.13187604075354, 4.373425028669592) 
  (74.39836054584774, 5.535116051909953) 
  (82.66484505094192, 6.833476607296237)
 GC130: x_cost_fuel[65] = PWL ( x[65] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC131: x_cost_cut[65] = PWL ( x_cut[65] ) : (0, 0) 
  (6.258093412347177, 0.0391637331576631) 
  (12.51618682469435, 0.1566549326306526) 
  (18.77428023704153, 0.3524735984189682) 
  (25.03237364938871, 0.6266197305226102) 
  (31.29046706173589, 0.9790933289415786) 
  (37.54856047408306, 1.409894393675873) 
  (43.80665388643024, 1.919022924725494) 
  (50.06474729877742, 2.506478922090441) 
  (56.32284071112459, 3.172262385770714) 
  (62.58093412347178, 3.916373315766315)
 GC132: x_cost_fuel[66] = PWL ( x[66] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC133: x_cost_cut[66] = PWL ( x_cut[66] ) : (0, 0) 
  (3.735567075234511, 0.0139544613735761) 
  (7.471134150469021, 0.0558178454943045) 
  (11.20670122570353, 0.125590152362185) 
  (14.94226830093804, 0.2232713819772179) 
  (18.67783537617255, 0.3488615343394029) 
  (22.41340245140706, 0.5023606094487402) 
  (26.14896952664158, 0.6837686073052297) 
  (29.88453660187609, 0.8930855279088715) 
  (33.62010367711059, 1.130311371259665) 
  (37.35567075234511, 1.395446137357611)
 GC134: x_cost_fuel[67] = PWL ( x[67] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC135: x_cost_cut[67] = PWL ( x_cut[67] ) : (0, 0) 
  (2.577433562208772, 0.0066431637676002) 
  (5.154867124417543, 0.0265726550704008) 
  (7.732300686626314, 0.0597884739084018) 
  (10.30973424883509, 0.1062906202816032) 
  (12.88716781104386, 0.1660790941900049) 
  (15.46460137325263, 0.2391538956336071) 
  (18.0420349354614, 0.3255150246124097) 
  (20.61946849767017, 0.4251624811264127) 
  (23.19690205987894, 0.538096265175616) 
  (25.77433562208772, 0.6643163767600198)
 GC136: x_cost_fuel[68] = PWL ( x[68] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC137: x_cost_cut[68] = PWL ( x_cut[68] ) : (0, 0) 
  (1.287967427203721, 0.00165886) (2.575934854407443, 0.00663544037415109) 
  (3.863902281611164, 0.01492974084184) 
  (5.151869708814885, 0.0265417614966044) 
  (6.439837136018608, 0.0414715023384443) 
  (7.727804563222328, 0.0597189633673598) 
  (9.01577199042605, 0.0812841445833509) 
  (10.30373941762977, 0.1061670459864175) 
  (11.59170684483349, 0.1343676675765597) 
  (12.87967427203722, 0.1658860093537774)
 GC138: x_cost_fuel[69] = PWL ( x[69] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC139: x_cost_cut[69] = PWL ( x_cut[69] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC140: x_cost_fuel[70] = PWL ( x[70] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC141: x_cost_cut[70] = PWL ( x_cut[70] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC142: x_cost_fuel[71] = PWL ( x[71] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC143: x_cost_cut[71] = PWL ( x_cut[71] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC144: x_cost_fuel[72] = PWL ( x[72] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC145: x_cost_cut[72] = PWL ( x_cut[72] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC146: x_cost_fuel[73] = PWL ( x[73] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC147: x_cost_cut[73] = PWL ( x_cut[73] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC148: x_cost_fuel[74] = PWL ( x[74] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC149: x_cost_cut[74] = PWL ( x_cut[74] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC150: x_cost_fuel[75] = PWL ( x[75] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC151: x_cost_cut[75] = PWL ( x_cut[75] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC152: x_cost_fuel[76] = PWL ( x[76] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC153: x_cost_cut[76] = PWL ( x_cut[76] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC154: x_cost_fuel[77] = PWL ( x[77] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC155: x_cost_cut[77] = PWL ( x_cut[77] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC156: x_cost_fuel[78] = PWL ( x[78] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC157: x_cost_cut[78] = PWL ( x_cut[78] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC158: x_cost_fuel[79] = PWL ( x[79] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC159: x_cost_cut[79] = PWL ( x_cut[79] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC160: x_cost_fuel[80] = PWL ( x[80] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC161: x_cost_cut[80] = PWL ( x_cut[80] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC162: x_cost_fuel[81] = PWL ( x[81] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC163: x_cost_cut[81] = PWL ( x_cut[81] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC164: x_cost_fuel[82] = PWL ( x[82] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC165: x_cost_cut[82] = PWL ( x_cut[82] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC166: x_cost_fuel[83] = PWL ( x[83] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC167: x_cost_cut[83] = PWL ( x_cut[83] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC168: x_cost_fuel[84] = PWL ( x[84] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC169: x_cost_cut[84] = PWL ( x_cut[84] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC170: x_cost_fuel[85] = PWL ( x[85] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC171: x_cost_cut[85] = PWL ( x_cut[85] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC172: x_cost_fuel[86] = PWL ( x[86] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC173: x_cost_cut[86] = PWL ( x_cut[86] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC174: x_cost_fuel[87] = PWL ( x[87] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC175: x_cost_cut[87] = PWL ( x_cut[87] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC176: x_cost_fuel[88] = PWL ( x[88] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC177: x_cost_cut[88] = PWL ( x_cut[88] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC178: x_cost_fuel[89] = PWL ( x[89] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC179: x_cost_cut[89] = PWL ( x_cut[89] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC180: x_cost_fuel[90] = PWL ( x[90] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC181: x_cost_cut[90] = PWL ( x_cut[90] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC182: x_cost_fuel[91] = PWL ( x[91] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC183: x_cost_cut[91] = PWL ( x_cut[91] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC184: x_cost_fuel[92] = PWL ( x[92] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC185: x_cost_cut[92] = PWL ( x_cut[92] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC186: x_cost_fuel[93] = PWL ( x[93] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC187: x_cost_cut[93] = PWL ( x_cut[93] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC188: x_cost_fuel[94] = PWL ( x[94] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC189: x_cost_cut[94] = PWL ( x_cut[94] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
 GC190: x_cost_fuel[95] = PWL ( x[95] ) : (150, 24.809) (210, 47.309) 
  (270, 77.009) (330, 113.909) (390, 158.009) (450, 209.309) 
  (510, 267.809) (570, 333.509) (630, 406.409) (690, 486.509) 
  (750, 573.809)
 GC191: x_cost_cut[95] = PWL ( x_cut[95] ) : (0, 0) (1e-06, 1e-15) 
  (2e-06, 4e-15) (3e-06, 9e-15) (4e-06, 1.6e-14) (5e-06, 2.5e-14) 
  (6e-06, 3.6e-14) (7e-06, 4.9e-14) (8e-06, 6.4e-14) (9e-06, 8.1e-14) 
  (1e-05, 1e-13)
End
