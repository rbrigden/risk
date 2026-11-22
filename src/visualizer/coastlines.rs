//! Board-space geographic silhouettes and organic interior border parameters in simulator region order.
//!
//! Canvas dimensions: `1280.0 × 680.0`.
//! Each continent defines 6 clockwise coastal sectors (`sectors[i].last() == sectors[(i + 1) % 6][0]`)
//! plus organic interior curve parameters (`inner_ratios`, `spoke_bend`, `hub_bend`) and verified
//! pole-of-inaccessibility troop marker anchors for all 7 provinces (6 outer sectors + 1 inner hub).

pub(super) struct Coastline {
    pub center: (f32, f32),
    pub label: (f32, f32),
    pub sectors: [&'static [(f32, f32)]; 6],
    pub inner_ratios: [f32; 6],
    pub spoke_bend: [f32; 6],
    pub hub_bend: [f32; 6],
    pub markers: [(f32, f32); 6],
    pub hub_marker: (f32, f32),
}

/// Gentle quadratic Bezier normal offsets (in board pixels) for the 9 sea bridges `A..=I`.
/// Chosen so every sea lane arcs gracefully across open ocean with zero land intersections.
pub(super) const BRIDGE_CURVES: [f32; 9] = [
    -14.0, // A: (1, 11)  Quebec <-> Gaul (North Atlantic)
    -9.0,  // B: (8, 18)  Baltics <-> Persia (Eurasian Strait)
    14.0,  // C: (22, 32) Bahia <-> Guinea (South Atlantic)
    24.0,  // D: (29, 39) Horn <-> Kimberley (Indian Ocean)
    -8.0,  // E: (3, 21)  Veracruz <-> Darien (Caribbean)
    -8.0,  // F: (10, 28) Iberia <-> Maghreb (Gibraltar)
    10.0,  // G: (17, 35) Deccan <-> Papua (Indo-Pacific)
    22.0,  // H: (2, 33)  Dixie <-> Sahara (Mid-Atlantic)
    -24.0, // I: (9, 40)  Balkans <-> Pilbara (Suez-Indian Sea Lane)
];

pub(super) const COASTLINES: [Coastline; 6] = [
    // 0: NORTH AMERICA (0:Yukon, 1:Quebec, 2:Dixie, 3:Veracruz, 4:Sierra, 5:Alaska, 6:Heartland)
    Coastline {
        center: (218.0, 184.0),
        label: (212.0, 36.0),
        sectors: [
            // 0: Yukon (Arctic coast + Hudson Bay indentation)
            &[
                (126.0, 94.0),
                (154.0, 78.0),
                (184.0, 74.0),
                (214.0, 78.0),
                (236.0, 90.0),
                (254.0, 76.0),
                (284.0, 76.0),
                (314.0, 92.0),
            ],
            // 1: Quebec (Labrador -> Nova Scotia -> New England) [Bridge A]
            &[
                (314.0, 92.0),
                (340.0, 108.0),
                (360.0, 130.0),
                (368.0, 154.0),
                (356.0, 174.0),
                (342.0, 194.0),
            ],
            // 2: Dixie (Carolinas -> Florida peninsula -> Gulf) [Bridge H]
            &[
                (342.0, 194.0),
                (344.0, 214.0),
                (350.0, 238.0),
                (338.0, 258.0),
                (324.0, 250.0),
                (308.0, 254.0),
                (294.0, 266.0),
            ],
            // 3: Veracruz (Mexico -> Yucatan -> Panama isthmus) [Bridge E]
            &[
                (294.0, 266.0),
                (298.0, 286.0),
                (294.0, 308.0),
                (274.0, 316.0),
                (252.0, 308.0),
                (232.0, 294.0),
                (212.0, 276.0),
            ],
            // 4: Sierra (Baja -> California -> Cascadia)
            &[
                (212.0, 276.0),
                (186.0, 268.0),
                (160.0, 252.0),
                (140.0, 230.0),
                (124.0, 206.0),
                (116.0, 182.0),
            ],
            // 5: Alaska (BC -> Alaska peninsula -> Bering Strait)
            &[
                (116.0, 182.0),
                (98.0, 166.0),
                (72.0, 166.0),
                (50.0, 152.0),
                (56.0, 130.0),
                (76.0, 114.0),
                (102.0, 102.0),
                (126.0, 94.0),
            ],
        ],
        inner_ratios: [0.44, 0.43, 0.44, 0.45, 0.43, 0.43],
        spoke_bend: [10.0, -9.0, 8.0, -10.0, 9.0, -8.0],
        hub_bend: [5.0, -4.0, 5.0, -4.0, 5.0, -4.0],
        markers: [
            (200.4, 111.7),
            (314.2, 149.0),
            (307.8, 224.0),
            (248.0, 266.0),
            (178.7, 226.1),
            (126.5, 145.3),
        ],
        hub_marker: (229.0, 183.1),
    },
    // 1: EUROPE (7:Nordics, 8:Baltics, 9:Balkans, 10:Iberia, 11:Gaul, 12:Albion, 13:Bohemia)
    Coastline {
        center: (586.0, 174.0),
        label: (586.0, 36.0),
        sectors: [
            // 7: Nordics (Scandinavian peninsula + Baltic Gulf)
            &[
                (512.0, 102.0),
                (528.0, 82.0),
                (552.0, 66.0),
                (578.0, 70.0),
                (596.0, 84.0),
                (618.0, 70.0),
                (648.0, 76.0),
                (674.0, 94.0),
            ],
            // 8: Baltics (Eastern European plain) [Bridge B]
            &[
                (674.0, 94.0),
                (698.0, 112.0),
                (714.0, 138.0),
                (718.0, 164.0),
                (708.0, 190.0),
            ],
            // 9: Balkans (Black Sea -> Aegean -> Adriatic) [Bridge I]
            &[
                (708.0, 190.0),
                (698.0, 218.0),
                (682.0, 244.0),
                (660.0, 262.0),
                (636.0, 268.0),
                (612.0, 258.0),
            ],
            // 10: Iberia (Italian coast -> Mediterranean -> Iberia) [Bridge F]
            &[
                (612.0, 258.0),
                (596.0, 274.0),
                (572.0, 278.0),
                (546.0, 272.0),
                (522.0, 274.0),
                (496.0, 258.0),
            ],
            // 11: Gaul (Bay of Biscay -> Brittany -> Channel) [Bridge A]
            &[
                (496.0, 258.0),
                (478.0, 238.0),
                (464.0, 212.0),
                (456.0, 186.0),
                (466.0, 162.0),
            ],
            // 12: Albion (British Isles bulge -> North Sea)
            &[
                (466.0, 162.0),
                (456.0, 140.0),
                (466.0, 116.0),
                (488.0, 104.0),
                (512.0, 102.0),
            ],
        ],
        inner_ratios: [0.43, 0.44, 0.43, 0.44, 0.43, 0.44],
        spoke_bend: [-8.0, 9.0, -8.0, 8.0, -9.0, 7.0],
        hub_bend: [-4.0, 5.0, -4.0, 4.0, -4.0, 4.0],
        markers: [
            (563.5, 104.5),
            (677.7, 145.5),
            (654.4, 217.9),
            (566.2, 243.0),
            (501.0, 202.2),
            (500.4, 132.6),
        ],
        hub_marker: (591.7, 173.3),
    },
    // 2: ASIA (14:Siberia, 15:Kamchatka, 16:Cathay, 17:Deccan, 18:Persia, 19:Urals, 20:Steppe)
    Coastline {
        center: (990.0, 190.0),
        label: (995.0, 36.0),
        sectors: [
            // 14: Siberia (Arctic coast)
            &[
                (850.0, 94.0),
                (890.0, 78.0),
                (936.0, 70.0),
                (986.0, 68.0),
                (1038.0, 72.0),
                (1088.0, 78.0),
                (1128.0, 92.0),
            ],
            // 15: Kamchatka (Chukotka -> Kamchatka -> Sea of Okhotsk)
            &[
                (1128.0, 92.0),
                (1168.0, 90.0),
                (1212.0, 100.0),
                (1218.0, 122.0),
                (1196.0, 142.0),
                (1186.0, 168.0),
                (1164.0, 192.0),
            ],
            // 16: Cathay (East China Sea -> Indochina peninsula)
            &[
                (1164.0, 192.0),
                (1164.0, 222.0),
                (1148.0, 254.0),
                (1128.0, 286.0),
                (1104.0, 308.0),
                (1080.0, 300.0),
                (1056.0, 284.0),
            ],
            // 17: Deccan (Bay of Bengal -> Indian subcontinent -> Arabian Sea) [Bridge G]
            &[
                (1056.0, 284.0),
                (1032.0, 304.0),
                (1006.0, 328.0),
                (982.0, 326.0),
                (958.0, 304.0),
                (928.0, 288.0),
            ],
            // 18: Persia (Arabia -> Levant -> Anatolia) [Bridge B]
            &[
                (928.0, 288.0),
                (896.0, 290.0),
                (862.0, 278.0),
                (828.0, 260.0),
                (798.0, 236.0),
                (782.0, 210.0),
                (778.0, 184.0),
            ],
            // 19: Urals (Ural foothills -> Western Siberia)
            &[
                (778.0, 184.0),
                (782.0, 156.0),
                (794.0, 130.0),
                (818.0, 108.0),
                (850.0, 94.0),
            ],
        ],
        inner_ratios: [0.43, 0.44, 0.44, 0.44, 0.44, 0.43],
        spoke_bend: [10.0, -10.0, 9.0, -9.0, 10.0, -9.0],
        hub_bend: [5.0, -5.0, 5.0, -5.0, 5.0, -5.0],
        markers: [
            (987.9, 110.4),
            (1145.3, 141.2),
            (1092.1, 248.2),
            (995.0, 279.8),
            (890.3, 237.2),
            (841.6, 144.3),
        ],
        hub_marker: (988.2, 194.5),
    },
    // 3: SOUTH AMERICA (21:Darien, 22:Bahia, 23:Pampas, 24:Patagonia, 25:Andes, 26:Peru, 27:Amazon)
    Coastline {
        center: (276.0, 490.0),
        label: (274.0, 658.0),
        sectors: [
            // 21: Darien (Panama -> Caribbean coast -> Orinoco) [Bridge E]
            &[
                (214.0, 394.0),
                (240.0, 376.0),
                (274.0, 368.0),
                (310.0, 372.0),
                (340.0, 386.0),
            ],
            // 22: Bahia (Northeast Brazil Atlantic bulge) [Bridge C]
            &[
                (340.0, 386.0),
                (370.0, 404.0),
                (396.0, 428.0),
                (402.0, 456.0),
                (392.0, 484.0),
                (374.0, 506.0),
            ],
            // 23: Pampas (Uruguay -> Rio de la Plata -> Argentina)
            &[
                (374.0, 506.0),
                (364.0, 536.0),
                (348.0, 566.0),
                (330.0, 592.0),
            ],
            // 24: Patagonia (Southern Cone -> Tierra del Fuego)
            &[
                (330.0, 592.0),
                (312.0, 616.0),
                (288.0, 634.0),
                (266.0, 636.0),
                (246.0, 622.0),
                (234.0, 598.0),
                (226.0, 576.0),
            ],
            // 25: Andes (Pacific Chile coast)
            &[
                (226.0, 576.0),
                (210.0, 550.0),
                (194.0, 522.0),
                (180.0, 494.0),
            ],
            // 26: Peru (Arica bend -> Peru -> Ecuador)
            &[
                (180.0, 494.0),
                (166.0, 466.0),
                (164.0, 438.0),
                (178.0, 414.0),
                (196.0, 402.0),
                (214.0, 394.0),
            ],
        ],
        inner_ratios: [0.43, 0.44, 0.43, 0.44, 0.43, 0.43],
        spoke_bend: [-8.0, 9.0, -8.0, 8.0, -8.0, 8.0],
        hub_bend: [-4.0, 4.0, -4.0, 4.0, -4.0, 4.0],
        markers: [
            (278.1, 405.7),
            (357.0, 454.3),
            (334.3, 533.9),
            (276.8, 589.7),
            (226.3, 530.6),
            (201.0, 452.7),
        ],
        hub_marker: (276.3, 490.9),
    },
    // 4: AFRICA (28:Maghreb, 29:Horn, 30:Zambezi, 31:Cape, 32:Guinea, 33:Sahara, 34:Congo)
    Coastline {
        center: (608.0, 474.0),
        label: (608.0, 654.0),
        sectors: [
            // 28: Maghreb (Atlas -> Tunisia -> Nile Delta) [Bridge F]
            &[
                (528.0, 366.0),
                (558.0, 348.0),
                (590.0, 342.0),
                (624.0, 346.0),
                (656.0, 354.0),
                (684.0, 368.0),
            ],
            // 29: Horn (Red Sea -> Horn of Africa) [Bridge D]
            &[
                (684.0, 368.0),
                (708.0, 392.0),
                (734.0, 418.0),
                (748.0, 444.0),
                (730.0, 470.0),
                (710.0, 492.0),
            ],
            // 30: Zambezi (Swahili coast -> Mozambique Channel)
            &[
                (710.0, 492.0),
                (702.0, 522.0),
                (692.0, 552.0),
                (676.0, 580.0),
            ],
            // 31: Cape (Southern Africa -> Cape of Good Hope)
            &[
                (676.0, 580.0),
                (654.0, 606.0),
                (626.0, 622.0),
                (598.0, 624.0),
                (572.0, 610.0),
                (552.0, 586.0),
            ],
            // 32: Guinea (Namib -> concave Gulf of Guinea) [Bridge C]
            &[
                (552.0, 586.0),
                (534.0, 556.0),
                (518.0, 524.0),
                (496.0, 498.0),
                (484.0, 474.0),
            ],
            // 33: Sahara (Senegal -> Mauritania -> Atlas) [Bridge H]
            &[
                (484.0, 474.0),
                (474.0, 446.0),
                (476.0, 418.0),
                (490.0, 392.0),
                (508.0, 376.0),
                (528.0, 366.0),
            ],
        ],
        inner_ratios: [0.43, 0.44, 0.44, 0.43, 0.44, 0.43],
        spoke_bend: [9.0, -9.0, 8.0, -8.0, 9.0, -8.0],
        hub_bend: [4.0, -4.0, 5.0, -4.0, 4.0, -4.0],
        markers: [
            (606.2, 387.5),
            (693.6, 440.6),
            (675.4, 518.3),
            (613.5, 573.6),
            (546.1, 512.2),
            (521.0, 428.5),
        ],
        hub_marker: (607.2, 473.3),
    },
    // 5: OCEANIA (35:Papua, 36:Coral Coast, 37:Tasman, 38:Outback, 39:Kimberley, 40:Pilbara, 41:Uluru)
    Coastline {
        center: (1056.0, 514.0),
        label: (1058.0, 656.0),
        sectors: [
            // 35: Papua (Arnhem -> Gulf of Carpentaria -> Cape York) [Bridge G]
            &[
                (976.0, 420.0),
                (1004.0, 400.0),
                (1034.0, 396.0),
                (1058.0, 412.0),
                (1084.0, 394.0),
                (1110.0, 396.0),
                (1130.0, 414.0),
            ],
            // 36: Coral Coast (Queensland -> Great Barrier Reef)
            &[
                (1130.0, 414.0),
                (1158.0, 434.0),
                (1184.0, 462.0),
                (1200.0, 490.0),
                (1198.0, 518.0),
            ],
            // 37: Tasman (NSW -> Victoria -> Tasman Sea)
            &[
                (1198.0, 518.0),
                (1192.0, 548.0),
                (1176.0, 578.0),
                (1152.0, 606.0),
                (1126.0, 618.0),
            ],
            // 38: Outback (Great Australian Bight concave arch)
            &[
                (1126.0, 618.0),
                (1094.0, 626.0),
                (1058.0, 616.0),
                (1022.0, 624.0),
                (990.0, 614.0),
            ],
            // 39: Kimberley (SW Australia -> Perth -> Shark Bay) [Bridge D]
            &[
                (990.0, 614.0),
                (956.0, 600.0),
                (930.0, 578.0),
                (916.0, 548.0),
                (912.0, 516.0),
            ],
            // 40: Pilbara (NW Cape -> Broome -> Timor Sea) [Bridge I]
            &[
                (912.0, 516.0),
                (912.0, 486.0),
                (924.0, 458.0),
                (948.0, 436.0),
                (976.0, 420.0),
            ],
        ],
        inner_ratios: [0.44, 0.43, 0.44, 0.44, 0.43, 0.44],
        spoke_bend: [-9.0, 9.0, -8.0, 9.0, -8.0, 8.0],
        hub_bend: [-4.0, 4.0, -5.0, 4.0, -4.0, 4.0],
        markers: [
            (1029.9, 435.6),
            (1142.3, 472.3),
            (1145.0, 552.1),
            (1052.0, 585.8),
            (966.5, 553.4),
            (964.3, 471.3),
        ],
        hub_marker: (1055.2, 511.3),
    },
];
