'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const core = require('../sites/amodal-preference/dist/core.js');
const study = {study_id:'test', round:'test-round', slots:['A','B','C'], cases:[{id:'P0001'}]};
const fixture = () => ({
  schema: core.SCHEMA, study_id:study.study_id, round:study.round, complete:true,
  reviewer:{code:'TEST_ONLY',role:'researcher',prior_exposure:'yes'}, exported_at:new Date().toISOString(),
  records:[{case_id:'P0001',status:'rated',judgements:{overall:{kind:'preference',winners:['A','C']},shape:null,surface:null},
    confidence:'',note:'=SUM(A1:A3)',skip_reason:'',created_at:'2026-11-22T00:00:00Z',updated_at:'2026-11-22T00:00:00Z'}]
});
test('export/import retains a partial top tie without method identities', () => {
  const validated = core.validate(JSON.parse(JSON.stringify(fixture())), study);
  assert.deepEqual(validated.records[0].judgements.overall.winners, ['A','C']);
  assert.equal(validated.complete, true);
  assert.match(core.csv(validated), /"'=SUM/);
});
test('wrong round, duplicate cases, unknown slots and fabricated completion are rejected', () => {
  for (const change of [
    r=>r.round='elsewhere', r=>r.records.push(r.records[0]),
    r=>r.records[0].judgements.overall.winners=['D'], r=>r.complete=false,
    r=>r.records[0].judgements.overall.winners=['A','A'],
    r=>r.records[0].status='skipped'
  ]) { const value=fixture(); change(value); assert.throws(()=>core.validate(value,study)); }
});
test('draft, skip, inability to judge, and tie remain distinct', () => {
  const value=fixture(); value.records[0].status='draft'; value.complete=false;
  value.records[0].judgements.overall=null;
  assert.equal(core.validate(value,study).complete,false);
  value.records[0].status='rated'; value.complete=true;
  assert.throws(()=>core.validate(value,study));
  value.records[0].judgements.overall={kind:'unassessable',winners:[]};
  assert.equal(core.validate(value,study).records[0].judgements.overall.kind,'unassessable');
  value.records[0].status='skipped'; value.records[0].skip_reason='test skip';
  assert.throws(()=>core.validate(value,study));
  value.records[0].judgements.overall=null;
  assert.equal(core.validate(value,study).records[0].status,'skipped');
});
