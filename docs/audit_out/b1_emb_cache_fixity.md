# B1 frozen-WiSig 512-D segment-embedding cache — fixity

encoder: runs/wisig_supcon_fft64/retrain_best/best_model.pt (frozen, read-only)
stage: get_encoder_output  dim: 512  stft: nfft=64 hop=16 (stft_for(256))
per segment: forward 30 windows -> mean-pool -> L2-normalize
throughput: 47,043 win/s, 1,568 seg/s, GPU 0.6377 ms/segment; cache 1798.7 MB

| collection | N | size MB | sha256 |
|---|---|---|---|
| Wired_indoors__Ch14_R1 | 73429 | 150.53 | 4dd6b10cf5d999cd669d06d440dd6f925f4b881ab31b239b9006bb399049f410 |
| Wired_indoors__Ch1_R1 | 73984 | 151.67 | d37fd877d18658ec63d18655ba413d4a9fe688f346c7d3904f631f79caf99cc2 |
| Wired_indoors__Ch1_R2 | 74237 | 152.19 | b158e681a04442d8dbad17f9112d64bb25b806b60afa48741361ea0485fa9319 |
| Wired_indoors__Ch2_R1 | 74070 | 151.84 | c186eb3f2659ab4c1fbb383a011496ce01bd5a59f06dae81e4845693741d59f1 |
| Wired_indoors__Ch32_R1 | 74400 | 152.52 | 17249f6936a357374673e2cbeab41055cc79553b36867bd6b752b06180ca0e1b |
| Wireless_Indoors__Ch2 | 74400 | 152.52 | 3cc761c7006c4f6904420071a6785429d57d6a2e5065662005f3af974970313c |
| Wireless_Indoors__R1 | 74400 | 152.52 | 062b912edc0af3f90623db7f0c5cc0a1a08b9e2283137bcc821cd8e64d992ce7 |
| Wireless_Indoors__R2 | 74031 | 151.76 | 4966f340da2201d46823e0a658b30b78cfb3bc73f0643bb5bdec8eb943c5c59a |
| Wireless_outdoors__Loc1 | 73064 | 149.78 | 5a25af5d40270193d1432242da91e694f5fe209e6a4aff1696dd2144a3d9f80e |
| Wireless_outdoors__Loc2 | 72093 | 147.79 | 8129ff5e250e14f97d47b981f55fb2c8d545b345b6b1197fce36e35fc8ca16a3 |
| Wireless_outdoors__Loc3 | 67745 | 138.88 | e1089ffc977d7f7b2147cd46bae4667feb225b9bdf712ed2a22a5270b426d1cf |
| Wireless_outdoors__Loc4 | 71565 | 146.71 | 1855dc730027cdd19b351d30254c1bb049ccf9af4489edca2fb9a6218a3a13a4 |
