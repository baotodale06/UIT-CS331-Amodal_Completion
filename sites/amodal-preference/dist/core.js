(function (root) {
  'use strict';
  const SCHEMA = 'amodal-preference-response-v1';
  const CRITERIA = ['overall', 'shape', 'surface'];
  function assert(ok, message) { if (!ok) throw new Error(message); }
  function judgement(value, slots, required) {
    if (value == null) { assert(!required, 'An overall judgement is required.'); return null; }
    assert(value && ['preference', 'unassessable'].includes(value.kind), 'Invalid judgement.');
    assert(Array.isArray(value.winners), 'Invalid option list.');
    const winners = value.winners;
    assert(new Set(winners).size === winners.length && winners.every(x => slots.includes(x)), 'Unknown or duplicate option.');
    assert(value.kind === 'preference' ? winners.length > 0 : winners.length === 0, 'Invalid tie or unassessable judgement.');
    return {kind: value.kind, winners: [...winners].sort()};
  }
  function record(value, study) {
    assert(value && study.cases.some(c => c.id === value.case_id), 'Unknown case.');
    assert(['rated', 'skipped', 'draft'].includes(value.status), 'Invalid case status.');
    assert(typeof value.note === 'string' && value.note.length <= 4000, 'Invalid note.');
    assert(['', 'low', 'medium', 'high'].includes(value.confidence), 'Invalid confidence.');
    const scores = {};
    assert(value.judgements && typeof value.judgements === 'object', 'Missing judgements.');
    for (const criterion of CRITERIA) scores[criterion] = judgement(value.judgements[criterion], study.slots, criterion === 'overall' && value.status === 'rated');
    const reason = value.skip_reason;
    assert(typeof reason === 'string' && reason.length <= 1000, 'Invalid skip reason.');
    if (value.status === 'skipped') {
      assert(reason.trim().length > 0, 'Please give a reason for skipping.');
      assert(CRITERIA.every(c => scores[c] === null), 'A skipped case cannot contain ratings.');
    }
    for (const field of ['created_at', 'updated_at']) assert(typeof value[field] === 'string' && Number.isFinite(Date.parse(value[field])), 'Invalid timestamp.');
    return {case_id: value.case_id, status: value.status, judgements: scores, confidence: value.confidence,
      note: value.note, skip_reason: reason, created_at: value.created_at, updated_at: value.updated_at};
  }
  function reviewer(value, required) {
    assert(value && typeof value.code === 'string' && value.code.length <= 80, 'Invalid rater code.');
    assert(['', 'researcher', 'independent', 'other'].includes(value.role), 'Invalid rater role.');
    assert(['', 'yes', 'no'].includes(value.prior_exposure), 'Invalid prior-exposure answer.');
    if (required) assert(value.code.trim() && value.role && value.prior_exposure, 'Complete your rater details before saving.');
    return {code: value.code.trim(), role: value.role, prior_exposure: value.prior_exposure};
  }
  function validate(value, study) {
    assert(value && value.schema === SCHEMA, 'This is not a preference-study backup.');
    assert(value.study_id === study.study_id && value.round === study.round, 'This backup belongs to a different round or set of images.');
    assert(Array.isArray(value.records), 'Missing case records.');
    const records = value.records.map(r => record(r, study));
    assert(new Set(records.map(r => r.case_id)).size === records.length, 'Duplicate case records.');
    const who = reviewer(value.reviewer, records.some(r => r.status !== 'draft'));
    const complete = records.filter(r => r.status !== 'draft').length === study.cases.length;
    assert(typeof value.complete === 'boolean' && value.complete === complete, 'Completion flag disagrees with the records.');
    return {schema: SCHEMA, study_id: study.study_id, round: study.round, reviewer: who, complete,
      exported_at: value.exported_at, records};
  }
  function csv(value) {
    const cell = x => '"' + String(x ?? '').replace(/^[=+@\-\t\r]/, "'$&").replaceAll('"', '""') + '"';
    const fields = ['study_id','round','rater_code','rater_role','prior_exposure','case_id','status','overall','shape','surface','confidence','note','skip_reason','updated_at'];
    const lines = value.records.map(r => [value.study_id,value.round,value.reviewer.code,value.reviewer.role,value.reviewer.prior_exposure,r.case_id,r.status,
      ...CRITERIA.map(c => r.judgements[c] ? (r.judgements[c].kind === 'unassessable' ? 'unassessable' : r.judgements[c].winners.join('=')) : ''),r.confidence,r.note,r.skip_reason,r.updated_at]);
    return '\ufeff' + [fields, ...lines].map(row => row.map(cell).join(',')).join('\r\n');
  }
  const api = {SCHEMA, CRITERIA, record, reviewer, validate, csv};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.PreferenceCore = api;
})(typeof window === 'undefined' ? globalThis : window);
