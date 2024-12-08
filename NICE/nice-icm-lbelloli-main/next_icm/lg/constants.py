_icm_lg_event_id = {
    'HSTD': 10,
    'HDVT': 20,
    'LSGS': 30,
    'LSGD': 40,
    'LDGS': 60,
    'LDGD': 50
}

_icm_lg_concatenation_event = 2014.0

_lg_matlab_event_id_map = [0, 30, 40, 60, 50, 10, 20]

_arduino_trigger_map = {
    0x40: 'HSTD',  # 64
    0x48: 'HDVT',  # 72
    0x80: 'LSGS',  # 128
    0x98: 'LDGD',  # 152
    0x90: 'LSGD',  # 144
    0x88: 'LDGS',  # 136

    # Inverted tones
    0x60: 'HSTD',  # 96
    0x68: 'HDVT',  # 104
    0xA0: 'LSGS',  # 160
    0xB8: 'LDGD',  # 184
    0xB0: 'LSGD',  # 176
    0xA8: 'LDGS'  # 168
}


_gtec_trig_map = {4: 64, 5: 72, 8: 128, 9: 136, 10: 144, 11: 152}
