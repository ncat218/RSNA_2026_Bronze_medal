"""Apply full CPU41 worker/controller tests to the symmetric42 geometry."""
from pathlib import Path

ROOT=Path(__file__).resolve().parent
test=(ROOT/'validate_41_crop_tta.py').read_text(encoding='utf-8')
test=test.replace('41_Anchor0958_S6_TwoCrop_Fixed10_Rank_Submit.ipynb',
                  '42_Anchor0958_S6_SymmetricCrop_Fixed10_Rank_Submit.ipynb')
test=test.replace('41_Build_Manifest.json','42_Build_Manifest.json')
test=test.replace('41_Local_Validation.json','42_Local_Validation.json')
test=test.replace('41_worker_patch_audit.json','42_worker_patch_audit.json')
test=test.replace('images[None,:,:,16:304,16:304]','images[None,:,:,8:296,8:296]')
test=test.replace('crop_offsets = ((16, 16), (24, 24))','crop_offsets = ((8, 8), (24, 24))')
test=test.replace('offset16/24','offset8/24')
exec(compile(test,'<42 full CPU validation>','exec'),{'__file__':str(ROOT/'validate_41_crop_tta.py'),'__name__':'__main__'})
