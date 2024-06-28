import { ChatPromptTemplate } from "@langchain/core/prompts";
import { InMemoryChatMessageHistory } from "@langchain/core/chat_history";
import {
  RunnablePassthrough,
  RunnableSequence,
  RunnableWithMessageHistory,
} from "@langchain/core/runnables";
import { ChatGroq } from "@langchain/groq";
import { AIMessage, BaseMessage, HumanMessage } from "@langchain/core/messages";

const model = new ChatGroq({
  model: "mixtral-8x7b-32768",
  temperature: 0,
});

const systemTemplate =
  "You are a helpful assistant who remembers all details the user shares with you.";
const promptTemplate = ChatPromptTemplate.fromMessages([
  ["system", systemTemplate],
  ["placeholder", "{chat_history}"],
  ["human", "{input}"],
]);

const messageHistories: Record<string, InMemoryChatMessageHistory> = {};

const filterMessages = ({ chat_history }: { chat_history: BaseMessage[] }) => {
  return chat_history.slice(-10);
};

const chain = RunnableSequence.from([
  RunnablePassthrough.assign({
    chat_history: filterMessages,
  }),
  promptTemplate,
  model,
]);

const messages = [
  new HumanMessage({ content: "hi! I'm bob" }),
  new AIMessage({ content: "hi!" }),
  new HumanMessage({ content: "I like vanilla ice cream" }),
  new AIMessage({ content: "nice" }),
  new HumanMessage({ content: "whats 2 + 2" }),
  new AIMessage({ content: "4" }),
  new HumanMessage({ content: "thanks" }),
  new AIMessage({ content: "No problem!" }),
  new HumanMessage({ content: "having fun?" }),
  new AIMessage({ content: "yes!" }),
  new HumanMessage({ content: "That's great!" }),
  new AIMessage({ content: "yes it is!" }),
];

const withMessageHistory = new RunnableWithMessageHistory({
  runnable: chain,
  getMessageHistory: async (sessionId) => {
    if (messageHistories[sessionId] === undefined) {
      const messageHistory = new InMemoryChatMessageHistory();
      await messageHistory.addMessages(messages);
      messageHistories[sessionId] = messageHistory;
    }
    return messageHistories[sessionId];
  },
  inputMessagesKey: "input",
  historyMessagesKey: "chat_history",
});

const config = {
  configurable: {
    sessionId: "abc2",
  },
};

const stream = await withMessageHistory.stream(
  {
    input: "whats my name?",
  },
  config,
);

for await (const chunk of stream) {
  console.log("|", chunk.content);
}
