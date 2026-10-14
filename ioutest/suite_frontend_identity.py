"""Narrow, audited cache compatibility for the text-padding validation fix."""

from pathlib import Path

from .common import digest, file_digest, read_json, write_json
from .suite_io import bind_identity


LEGACY_SOURCE_SHA256 = '860a190972e8b7561a3a7d69028e2f67cc408112066bee43b68c1d784d30a7a5'
FIXED_SOURCE_SHA256 = 'eb992e5e33a5207844ec48076ce25299af093adc1149e9bfa0c4d5df199cafeb'
TEXT_LOGITS_POLICY = 'finite-active-tokens; allow-native-negative-infinity-padding'


def bind_frontend_identity(path, identity, component):
    """Reuse unchanged GT/depth only for this exact, verified source transition."""
    path = Path(path)
    if not path.exists():
        return bind_identity(path, identity)
    old = read_json(path)
    if old == dict(signature=digest(identity), provenance=identity):
        return bind_identity(path, identity)
    legacy_identity = dict(identity, worker_sha256=LEGACY_SOURCE_SHA256)
    if component == 'text':
        legacy_identity.pop('grounding_logits_policy', None)
    compatible = (component in ('gt-box', 'depth', 'text') and
                  (component != 'text' or identity.get('grounding_logits_policy') == TEXT_LOGITS_POLICY) and
                  identity['worker_sha256'] == FIXED_SOURCE_SHA256 and
                  file_digest(Path(__file__).with_name('suite_frontend.py')) == FIXED_SOURCE_SHA256 and
                  old == dict(signature=digest(legacy_identity), provenance=legacy_identity))
    if not compatible:
        return bind_identity(path, identity)  # Standard immutable-cache rejection.
    audit = dict(component=component, old_source_sha256=LEGACY_SOURCE_SHA256,
                 current_source_sha256=FIXED_SOURCE_SHA256,
                 old_signature=old['signature'], requested_signature=digest(identity),
                 compatibility_checker_sha256=file_digest(__file__))
    if component == 'text':
        # The defective validator aborted before creating any text frame output.
        # A nonempty text run must retain normal strict cache protection.
        if any((path.parent/'frames').rglob('*.json.gz')):
            return bind_identity(path, identity)
        archive = path.with_name('model.before-text-padding-fix.json')
        if archive.exists() and read_json(archive) != old:
            raise ValueError('已有文本前处理来源归档与原记录不同')
        write_json(archive, old)
        audit['action'] = 'replace-empty-text-identity; no predictions reused'
        write_json(path.with_name('text_padding_fix_compatibility.json'), audit)
        return bind_identity_replacing_empty(path, identity)
    # This patch changes only the text validation path. SAM2 GT-box and MoGe
    # depth computations are unchanged; retain their original signatures so
    # the already completed object predictions remain valid. Record the actual
    # new source and decision separately, without rewriting original provenance.
    audit['action'] = 'retain-original-signature; GT-mask/depth-computation-unchanged'
    write_json(path.with_name('text_padding_fix_compatibility.json'), audit)
    return old['signature']


def bind_identity_replacing_empty(path, identity):
    signature = digest(identity)
    write_json(path, dict(signature=signature, provenance=identity))
    return signature
