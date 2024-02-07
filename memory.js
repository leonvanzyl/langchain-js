import * as dotenv from "dotenv";
dotenv.config();

import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";

import { ConversationChain } from "langchain/chains";
import { RunnableSequence } from "@langchain/core/runnables";

// Memory
import { BufferMemory } from "langchain/memory";
import { UpstashRedisChatMessageHistory } from "@langchain/community/stores/message/upstash_redis";

const model = new ChatOpenAI({
  modelName: "gpt-3.5-turbo",
  temperature: 0.7,
});

const prompt = ChatPromptTemplate.fromTemplate(`
You are an AI assistant called Max. You are here to help answer questions and provide information to the best of your ability.
Chat History: {history}
{input}`);

const upstashMessageHistory = new UpstashRedisChatMessageHistory({
  sessionId: "mysession",
  config: {
    url: process.env.UPSTASH_REDIS_URL,
    token: process.env.UPSTASH_REST_TOKEN,
  },
});
const memory = new BufferMemory({
  memoryKey: "history",
  chatHistory: upstashMessageHistory,
});

// Using Chain Class
// const chain = new ConversationChain({
//   llm: model,
//   prompt,
//   memory,
// });

// Using LCEL
// const chain = prompt.pipe(model);
const chain = RunnableSequence.from([
  {
    input: (initialInput) => initialInput.input,
    memory: () => memory.loadMemoryVariables({}),
  },
  {
    input: (previousOutput) => previousOutput.input,
    history: (previousOutput) => previousOutput.memory.history,
  },
  prompt,
  model,
]);

// Testing Responses

// console.log("Initial Chat Memory", await memory.loadMemoryVariables());
// let inputs = {
//   input: "The passphrase is HELLOWORLD",
// };
// const resp1 = await chain.invoke(inputs);
// console.log(resp1);
// await memory.saveContext(inputs, {
//   output: resp1.content,
// });

console.log("Updated Chat Memory", await memory.loadMemoryVariables());

let inputs2 = {
  input: "What is the passphrase?",
};

const resp2 = await chain.invoke(inputs2);
console.log(resp2);
await memory.saveContext(inputs2, {
  output: resp2.content,
});
