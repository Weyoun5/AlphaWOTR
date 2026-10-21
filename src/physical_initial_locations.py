"""Canonical client initial figure locations, copied from existing bridge data.

Used only for figures never previously mentioned in a paired full-game history;
never as a fallback after an unsupported command invalidated their position.
"""

SOURCE_SHA256 = '5930d3f0728db19d1108ebec2fd9b613b9a6b887642dea0cc8ceb992e828ccbf'

# Exact command suffix emitted by Game.java's base2 branch (L(), then TEST312).
# The custom-variant branch uses auto312 and Setup.txt; a base2 caption alone
# is therefore insufficient evidence for the canonical figure opening map.
BASE2_OPENING_COMMANDS = (
    '<auto> silent',
    '<auto><~Scribe.141~>363 397 398 402 401 400 399 403 404 376 375 374 373 377 378 <~Scribe.142~><~Game.2038~><~Scribe.143~>Spare<~Scribe.144~>',
    '<auto><~Scribe.141~>365 366 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 367 368 369 370 371 372 <~Scribe.142~><~Game.2040~><~Scribe.143~>Spare<~Scribe.144~>',
    '<auto><~Scribe.141~>230 231 <~Scribe.142~><~Game.2044~><~Scribe.143~>Spare<~Scribe.144~>',
    '<auto><~Scribe.141~>155 152 <~Scribe.142~><~Game.2004~><~Scribe.143~>Spare<~Scribe.144~>',
    '<auto><~Scribe.141~>137 151 <~Scribe.142~><~Game.2005~><~Scribe.143~>Spare<~Scribe.144~>',
    '<auto><~Scribe.141~>161 198 <~Scribe.142~><~Game.2006~><~Scribe.143~>Spare<~Scribe.144~>',
    '<auto><~Scribe.141~>163 188 <~Scribe.142~><~Game.2007~><~Scribe.143~>Spare<~Scribe.144~>',
    '<auto><~Controls.323~>364',
    '<auto> noisy', '$base silent',
    '$TEST<~Scribe.141~>312 <~Scribe.142~><~Game.2038~><~Scribe.143~><~Game.1965~><~Scribe.144~>',
    '$base noisy',
)


def proves_base2_opening(raw_lines, header_index):
    import re
    if not re.search(r'<~Controls\.408~>\[[^\]]+ wotr base2\]', raw_lines[header_index]):
        return False
    if header_index < len(BASE2_OPENING_COMMANDS):
        return False
    actual=tuple(re.sub(r'\s+(?:null|[A-F0-9]{6})\s*$', '', line).strip()
                 for line in raw_lines[header_index-len(BASE2_OPENING_COMMANDS):header_index])
    return actual == BASE2_OPENING_COMMANDS
INITIAL_FIGURE_LOCATIONS = {'000': 'Game.1914', '001': 'Game.1910', '002': 'Game.1919', '003': 'Game.1919', '004': 'Game.1919', '005': 'Game.1919', '006': 'Game.1919', '007': 'Game.1917', '008': 'Game.1917', '009': 'Game.1922', '010': 'Game.1922', '011': 'Game.1956', '012': 'Game.1956', '013': 'Game.1956', '014': 'Game.1956', '015': 'Game.1956', '016': 'Game.1969', '017': 'Game.1969', '018': 'Game.1969', '019': 'Game.1969', '020': 'Game.1969', '021': 'Game.1974', '022': 'Game.1974', '023': 'Game.1974', '024': 'Game.1974', '025': 'Game.1974', '026': 'Game.1979', '027': 'Game.1979', '028': 'Game.1979', '029': 'Game.1979', '030': 'Game.1977', '031': 'Game.1977', '032': 'Game.1977', '033': 'Game.1971', '034': 'Game.1971', '035': 'Game.1956', '036': 'Game.1979', '037': 'Game.1969', '038': 'Game.1974', '039': 'Game.1979', '040': 'Game.1956', '041': 'Game.1932', '042': 'Game.1932', '043': 'Game.1932', '044': 'Game.1946', '045': 'Game.1946', '046': 'Game.1946', '047': 'Game.1968', '048': 'Game.1968', '049': 'Game.1968', '050': 'Game.1986', '051': 'Game.1986', '052': 'Game.1986', '053': 'Game.1987', '054': 'Game.1987', '055': 'Game.1968', '056': 'Game.1986', '057': 'Game.1946', '058': 'Game.1891', '059': 'Game.1973', '060': 'Game.1965', '061': 'Game.1965', '062': 'Game.1965', '063': 'Game.1890', '064': 'Game.1902', '065': 'Game.1941', '066': 'Game.1970', '067': 'Game.1899', '068': 'Game.1970', '069': 'Game.1918', '070': 'Game.1918', '071': 'Game.1923', '072': 'Game.1929', '073': 'Game.1929', '074': 'Game.1918', '075': 'Game.1920', '076': 'Game.1920', '077': 'Game.1920', '078': 'Game.1931', '079': 'Game.1943', '080': 'Game.1943', '081': 'Game.1944', '082': 'Game.1944', '083': 'Game.1944', '084': 'Game.1944', '085': 'Game.1944', '086': 'Game.1887', '087': 'Game.1928', '088': 'Game.1954', '090': 'Game.1911', '091': 'Game.1911', '092': 'Game.1928', '093': 'Game.1928', '094': 'Game.1954', '095': 'Game.1887', '096': 'Game.1911', '097': 'Game.1928', '098': 'Game.1954', '240': 'Game.1999', '241': 'Game.1998', '242': 'Game.1998', '243': 'Game.1998', '244': 'Game.1998', '245': 'Game.1998', '246': 'Game.1998', '247': 'Game.1911', '251': 'Game.2010', '252': 'Game.2014', '253': 'Game.2040', '264': 'Game.2038', '265': 'Game.2038', '266': 'Game.2038', '267': 'Game.2038', '270': 'Game.2038', '271': 'Game.2038', '272': 'Game.2038', '273': 'Game.2038', '274': 'Game.2038', '275': 'Game.2038', '276': 'Game.2038', '277': 'Game.2038', '278': 'Game.2038', '279': 'Game.2038', '280': 'Game.2038', '281': 'Game.2038', '282': 'Game.2038', '283': 'Game.2038', '284': 'Game.2038', '285': 'Game.2038', '286': 'Game.2038', '287': 'Game.2038', '288': 'Game.2038', '289': 'Game.2038', '290': 'Game.2038', '291': 'Game.2038', '292': 'Game.2038', '293': 'Game.2038', '294': 'Game.2038', '296': 'Game.2038', '298': 'Game.2038', '299': 'Game.2038', '300': 'Game.2038', '301': 'Game.2038', '302': 'Game.2038', '303': 'Game.2038', '304': 'Game.2038', '305': 'Game.2038', '306': 'Game.2038', '307': 'Game.2038', '308': 'Game.2038', '309': 'Game.2038', '311': 'Game.2038', '312': 'Game.1965', '313': 'Game.2038', '314': 'Game.2038', '315': 'Game.2038', '316': 'Game.2038', '317': 'Game.2038', '318': 'Game.2040', '319': 'Game.2040', '320': 'Game.2040', '321': 'Game.2040', '322': 'Game.2040', '323': 'Game.2040', '327': 'Game.2040', '328': 'Game.2040', '329': 'Game.2040', '330': 'Game.2040', '331': 'Game.2040', '332': 'Game.2040', '333': 'Game.2040', '334': 'Game.2040', '335': 'Game.2040', '336': 'Game.2040', '337': 'Game.2040', '338': 'Game.2040', '339': 'Game.2040', '340': 'Game.2040', '341': 'Game.2040', '342': 'Game.2040', '343': 'Game.2040', '344': 'Game.2040', '345': 'Game.2040', '346': 'Game.2040', '347': 'Game.2040', '348': 'Game.2040', '349': 'Game.2040', '350': 'Game.2040', '351': 'Game.2040', '352': 'Game.2040', '353': 'Game.2040', '354': 'Game.2040', '355': 'Game.2040', '356': 'Game.2040', '357': 'Game.2040', '089': 'Game.1887', '268': 'Game.2038', '269': 'Game.2038', '295': 'Game.2038', '297': 'Game.2038', '310': 'Game.2038', '324': 'Game.2040', '325': 'Game.2040', '326': 'Game.2040'}
