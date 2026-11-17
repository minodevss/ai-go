import { AbstractChat, DefaultChatTransport, TextStreamChatTransport, lastAssistantMessageIsCompleteWithApprovalResponses, lastAssistantMessageIsCompleteWithToolCalls, readUIMessageStream } from 'ai';

class State {
  status = 'ready';
  error = undefined;
  messages = [];
  pushMessage(m) { this.messages.push(m); }
  popMessage() { this.messages.pop(); }
  replaceMessage(i, m) { this.messages[i] = m; }
  snapshot(t) { return structuredClone(t); }
}

class Chat extends AbstractChat {
  constructor(options) { super({ ...options, state: new State() }); }
}

const [base, scenario, extra] = process.argv.slice(2);
const errors = [];
let counter = 0;
const generateId = () => `client-${++counter}`;

function newChat(path, options = {}) {
  return new Chat({
    id: 'chat-1',
    generateId,
    transport: new DefaultChatTransport({ api: `${base}${path}`, ...(options.transport ?? {}) }),
    onError: (e) => errors.push(String(e.message)),
    ...options.chat,
  });
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function settle(chat) {
  let quiet = 0;
  for (let i = 0; i < 400 && quiet < 5; i++) {
    await sleep(20);
    quiet = chat.status === 'ready' ? quiet + 1 : 0;
  }
}

function approvalParts(chat) {
  return chat.messages.flatMap((m) => m.parts).filter((p) => p.state === 'approval-requested');
}

const scenarios = {
  async text() {
    const chat = newChat('/text');
    await chat.sendMessage({ text: 'hi' });
    await settle(chat);
    return chat;
  },
  async history() {
    const chat = newChat('/history');
    await chat.sendMessage({ text: 'one' });
    await settle(chat);
    await chat.sendMessage({ text: 'two' });
    await settle(chat);
    return chat;
  },
  async regenerate() {
    const chat = newChat('/history');
    await chat.sendMessage({ text: 'one' });
    await settle(chat);
    await chat.regenerate();
    await settle(chat);
    return chat;
  },
  async approval() {
    const chat = newChat((extra ?? '').includes('signed') ? '/approval-signed' : '/approval', { chat: { sendAutomaticallyWhen: lastAssistantMessageIsCompleteWithApprovalResponses } });
    await chat.sendMessage({ text: 'delete it' });
    await settle(chat);
    const pending = approvalParts(chat);
    if (pending.length === 0) return chat;
    await chat.addToolApprovalResponse({ id: pending[0].approval.id, approved: !(extra ?? '').includes('deny'), reason: (extra ?? '').includes('deny') ? 'not now' : undefined });
    await settle(chat);
    return chat;
  },
  async clientTool() {
    const chat = newChat('/clienttool', {
      chat: {
        sendAutomaticallyWhen: lastAssistantMessageIsCompleteWithToolCalls,
        async onToolCall({ toolCall }) {
          void chat.addToolOutput({ tool: toolCall.toolName, toolCallId: toolCall.toolCallId, output: { city: 'Paris' } });
        },
      },
    });
    await chat.sendMessage({ text: 'where am I' });
    await settle(chat);
    return chat;
  },
  async file() {
    const chat = newChat('/file');
    await chat.sendMessage({ text: 'see', files: [{ type: 'file', mediaType: 'text/plain', filename: 'a.txt', url: 'data:text/plain;base64,aGVsbG8=' }] });
    await settle(chat);
    return chat;
  },
  async data() {
    const received = [];
    const chat = newChat('/data', { chat: { onData: (d) => received.push(d) } });
    await chat.sendMessage({ text: 'data' });
    await settle(chat);
    chat.received = received;
    return chat;
  },
  async error() {
    const chat = newChat('/error');
    await chat.sendMessage({ text: 'boom' });
    await settle(chat);
    return chat;
  },
  async resume() {
    const first = newChat('/resume');
    const sent = first.sendMessage({ text: 'long' });
    await sleep(200);
    await first.stop();
    await sent.catch(() => {});
    const second = newChat('/resume', { chat: { id: 'chat-1' } });
    await second.resumeStream();
    await settle(second);
    return second;
  },
  async resumeNone() {
    const chat = newChat('/resume');
    await chat.resumeStream();
    await settle(chat);
    return chat;
  },
  async textStream() {
    const chat = newChat('/plain', { transport: {} });
    chat.transport = new TextStreamChatTransport({ api: `${base}/plain` });
    await chat.sendMessage({ text: 'hi' });
    await settle(chat);
    return chat;
  },
  async readMessages() {
    const res = await fetch(`${base}/text`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ id: 'x', messages: [{ id: 'u1', role: 'user', parts: [{ type: 'text', text: 'hi' }] }], trigger: 'submit-message' }) });
    const transport = new DefaultChatTransport({ api: `${base}/text` });
    const chunks = transport.processResponseStream(res.body);
    let last;
    for await (const message of readUIMessageStream({ stream: chunks })) last = message;
    return { message: last };
  },
};

const run = scenarios[scenario];
if (!run) {
  console.error(`unknown scenario ${scenario}`);
  process.exit(2);
}
const result = await run();
if (result instanceof Chat) {
  console.log(JSON.stringify({ status: result.status, messages: result.messages, errors, received: result.received ?? [] }));
} else {
  console.log(JSON.stringify(result));
}
process.exit(0);
