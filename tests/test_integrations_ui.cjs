const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

function setup() {
  const nodes = new Map(), sent = [];
  class Element {
    constructor() { this.children = []; this.events = {}; this.value = ''; this.type = 'password'; }
    appendChild(child) { this.children.push(child); }
    replaceChildren() { this.children = []; }
    addEventListener(name, handler) { this.events[name] = handler; }
    setAttribute(key, value) { this[key] = value; if (key === 'id') nodes.set(value, this); }
  }
  const node = id => {
    if (!nodes.has(id)) nodes.set(id, new Element());
    return nodes.get(id);
  };
  const cfg = {enabled: false, auto_render: false, services: {
    radarr: {url:'http://127.0.0.1:7878', server_root:'/nas', local_root:'/Volumes/Public', tag:'cleancut', key_configured:true},
    sonarr: {url:'http://127.0.0.1:8989', server_root:'/nas', local_root:'/Volumes/Public', tag:'cleancut'},
  }};
  const context = vm.createContext({
    document: {getElementById: node, createElement: () => new Element()},
    setTimeout, clearTimeout, confirm: () => true,
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../static/app.js'), 'utf8'), context);
  context.api = async () => ({settings: cfg, events: []});
  context.post = async (url, body) => { sent.push({url, body}); return {token:'test-secret'}; };
  context.toast = () => {};
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../static/integrations.js'), 'utf8'), context);
  return {context, node, sent, run: code => vm.runInContext(code, context)};
}

test('load saved fields without exposing keys; save selected policy', async () => {
  const s = setup();
  await s.run('refresh(true)');
  assert.equal(s.node('radarr-api_key').value, '');
  assert.match(s.node('radarr-api_key').placeholder, /Saved/);
  s.node('integration-enabled').checked = true;
  s.node('integration-auto').checked = false;
  await s.node('save-integrations').onclick();
  assert.equal(s.sent[0].body.enabled, true);
  assert.equal(s.sent[0].body.auto_render, false);
  assert.equal(s.sent[0].body.services.radarr.server_root, '/nas');
});

test('enabling requires confirmation; showing/hiding token is explicit', async () => {
  const s = setup();
  await s.run('refresh(true)');
  s.context.confirm = () => false;
  s.node('integration-enabled').checked = true;
  await s.node('save-integrations').onclick();
  assert.equal(s.sent.length, 0);
  await s.node('show-webhook-token').onclick();
  assert.equal(s.node('webhook-token').value, 'test-secret');
  await s.node('show-webhook-token').onclick();
  assert.equal(s.node('webhook-token').value, '');
  assert.equal(s.node('webhook-token').type, 'password');
});

test('recover button uses journal id and never puts file paths in markup', async () => {
  const s = setup();
  await s.run('refresh(true)');
  s.run(`showImports([{id:8,service:'radarr',local_path:'<img src=x onerror=alert(1)>',
    state:'published',job_status:'failed',job_id:1,render_id:2,backup:'/nas/.backup',partial:'/nas/.upload'}]);`);
  const box = s.node('imports').children[0];
  assert.match(box.children[0].textContent, /<img/);
  assert.equal(box.children[0].innerHTML, undefined);
  const button = box.children.find(child => child.textContent === 'Recover / confirm rescan');
  await button.events.click({target: button});
  assert.equal(s.sent[0].url, '/api/integrations/8/recover');
});
