// Optional dependency-free Edge/Chrome CDP smoke; isolated QA study and profile only.
const fs = require('node:fs/promises');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const assert = require('node:assert/strict');
const [folder, port='9338'] = process.argv.slice(2);
if (!folder) throw new Error('Pass the isolated QA packet directory and CDP port.');
const delay = ms => new Promise(r=>setTimeout(r,ms));
(async()=>{
  const tabs = await (await fetch('http://127.0.0.1:'+port+'/json/list')).json();
  const ws = new WebSocket(tabs.find(t=>t.type==='page').webSocketDebuggerUrl);
  await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject;});
  let id=0; const pending=new Map(), errors=[];
  ws.onmessage=event=>{
    const message=JSON.parse(event.data);
    if (message.id) {
      const p=pending.get(message.id); pending.delete(message.id);
      if (message.error) p.reject(new Error(JSON.stringify(message.error))); else p.resolve(message.result);
    } else if (message.method==='Runtime.exceptionThrown') errors.push(message.params.exceptionDetails.text);
  };
  const call=(method,params={})=>new Promise((resolve,reject)=>{
    const request=++id; pending.set(request,{resolve,reject}); ws.send(JSON.stringify({id:request,method,params}));
  });
  const evaluate=async expression=>{
    const result=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});
    if(result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
    return result.result.value;
  };
  const waitFor=async expression=>{
    for(let i=0;i<80;i++){if(await evaluate(expression))return;await delay(100);}
    throw new Error('Timed out: '+expression);
  };
  const click=id=>evaluate('document.getElementById('+JSON.stringify(id)+').click()');
  try {
    await call('Page.enable'); await call('Runtime.enable');
    await call('Emulation.setDeviceMetricsOverride',{width:1440,height:1200,deviceScaleFactor:1,mobile:false});
    await call('Page.navigate',{url:pathToFileURL(path.join(folder,'index.html')).href});
    await waitFor("document.querySelectorAll('.candidate img').length===2 && [...document.querySelectorAll('.candidate img')].every(x=>x.naturalWidth>0) && !document.querySelector('.pick').disabled");
    assert.equal(await evaluate("Object.keys(localStorage).filter(k=>!k.includes('qa-')).length"),0);
    const screenshot=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:true});
    await fs.writeFile(path.join(folder,'..','desktop.png'),Buffer.from(screenshot.data,'base64'));
    await evaluate("document.querySelector('.pick').click()");
    await click('save-next');
    await waitFor("document.getElementById('rater-dialog').open");
    await evaluate("document.getElementById('rater-code').value='TEST_ONLY';document.getElementById('rater-role').value='researcher';document.getElementById('rater-exposure').value='yes';document.getElementById('rater-form').requestSubmit()");
    await waitFor("document.getElementById('case-position').textContent.includes('CASE 2 OF') && !document.querySelector('.pick').disabled");
    await evaluate("[...document.querySelectorAll('#overall-controls button')].find(b=>b.textContent==='All equal').click()");
    await click('save-next');
    await waitFor("document.getElementById('case-position').textContent.includes('CASE 3 OF')");
    await click('skip-button'); await evaluate("document.getElementById('skip-form').requestSubmit()");
    await waitFor("document.getElementById('finish-dialog').open");
    let response=await evaluate("JSON.parse(localStorage.getItem(Object.keys(localStorage)[0])).response");
    assert.equal(response.complete,true); assert.equal(response.records.length,3);
    assert.equal(response.records[0].judgements.overall.winners.length,1);
    assert.equal(response.records[1].judgements.overall.winners.length,2);
    assert.equal(response.records[2].status,'skipped');
    const downloadFolder=path.resolve(folder,'..','downloads'); await fs.mkdir(downloadFolder,{recursive:true});
    await call('Browser.setDownloadBehavior',{behavior:'allow',downloadPath:downloadFolder});
    await click('finish-export'); await delay(600);
    const downloaded=(await fs.readdir(downloadFolder)).find(n=>n.endsWith('.json'));
    assert.ok(downloaded,'JSON export download');
    const backup=JSON.parse(await fs.readFile(path.join(downloadFolder,downloaded),'utf8'));
    assert.equal(backup.study_id,response.study_id); assert.equal(backup.complete,true);
    await call('Page.reload');
    await waitFor("document.getElementById('progress-fraction').textContent==='3 / 3'");
    // The compact hosted interface must wait for the folder without failed image requests.
    await call('Page.addScriptToEvaluateOnNewDocument',{source:'window.PREFERENCE_LOCAL_IMAGES = true;'});
    await call('Page.reload');
    await waitFor("document.getElementById('asset-status').textContent.includes('Open study folder')");
    assert.equal(await evaluate("document.querySelector('.pick').disabled && document.getElementById('skip-button').disabled"),true);
    assert.equal(await evaluate("[...document.querySelectorAll('.candidate img')].every(x=>!x.hasAttribute('src'))"),true);
    // Mount the local folder as the hosted site's exact folder loader does.
    const document=await call('DOM.getDocument');
    const node=await call('DOM.querySelector',{nodeId:document.root.nodeId,selector:'#folder-input'});
    await call('DOM.setFileInputFiles',{nodeId:node.nodeId,files:[path.resolve(folder)]});
    await waitFor("document.getElementById('folder-status').textContent.includes('15 images ready')");
    await click('previous');
    await waitFor("!document.querySelector('.pick').disabled");
    await evaluate("window.confirm=()=>true");
    const importNode=await call('DOM.querySelector',{nodeId:document.root.nodeId,selector:'#import-file'});
    await call('DOM.setFileInputFiles',{nodeId:importNode.nodeId,files:[path.join(downloadFolder,downloaded)]});
    await waitFor("document.getElementById('notice').textContent.includes('Backup restored')");
    const invalid=structuredClone(backup); invalid.study_id='wrong-round';
    const invalidPath=path.join(downloadFolder,'wrong-round.json'); await fs.writeFile(invalidPath,JSON.stringify(invalid));
    await call('DOM.setFileInputFiles',{nodeId:importNode.nodeId,files:[invalidPath]});
    await waitFor("document.getElementById('notice').textContent.includes('not imported')");
    assert.equal(await evaluate("JSON.parse(localStorage.getItem(Object.keys(localStorage)[0])).response.records.length"),3);
    await call('Emulation.setDeviceMetricsOverride',{width:390,height:900,deviceScaleFactor:1,mobile:true});
    await delay(300);
    assert.equal(await evaluate('document.documentElement.scrollWidth<=window.innerWidth'),true,'mobile horizontal overflow');
    const mobile=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:true});
    await fs.writeFile(path.join(folder,'..','mobile.png'),Buffer.from(mobile.data,'base64'));
    assert.deepEqual(errors,[]);
    const receipt={status:'passed',cases:3,synthetic_responses_only:true,
      checks:['native images load','anonymous choice','rater disclosure','tie','skip','completion','local save/reload','JSON download','hosted folder-required state','folder mount and checksums','backup import','wrong-round rejection','390px responsive layout'],console_exceptions:errors};
    await fs.writeFile(path.join(folder,'..','browser-check.json'),JSON.stringify(receipt,null,2));
    console.log(JSON.stringify(receipt));
  } finally {await call('Browser.close').catch(()=>{}); ws.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
