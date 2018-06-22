\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  3 var0 + var1 + 3 var2 + var3 + var4 + 3 var5 + var6
Subject To
 constr0: var1 + var3 + var4 + var6 <= 3
 constr1: 3 var0 + var3 <= 3
 constr2: var3 + 3 var5 <= 3
 constr3: 3 var2 + var3 <= 3
 constr4: var0 + var1 + var2 + var5 <= 3
 constr5: var0 + var2 + var5 + var6 <= 3
 constr6: var0 + var2 + var4 + var5 <= 3
Bounds
Generals
 var0 var1 var2 var3 var4 var5 var6
End
