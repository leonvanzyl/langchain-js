import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import {
  CommaSeparatedListOutputParser,
  StringOutputParser,
  BaseOutputParser,
} from "@langchain/core/output_parsers";

// Import environment variables
import * as dotenv from "dotenv";
dotenv.config();

// Instantiate the model
const model = new ChatOpenAI({
  modelName: "gpt-3.5-turbo",
  temperature: 0.9,
});

// Create Prompt Template from fromMessages
const prompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    "Provide 5 synonyms, seperated by commas, for a word that the user will provide.",
  ],
  ["human", "{word}"],
]);

// Custom output parser
class MyParser extends BaseOutputParser {
  parse(output) {
    console.log("Custom Parser:", output);
    return output.split(",");
  }
}

// Create the output parser
// const outputParser = new CommaSeparatedListOutputParser();
const outputParser = new MyParser();

// Create the Chain
const chain = prompt.pipe(model).pipe(outputParser);

const response = await chain.invoke({
  word: "chicken",
});

console.log(response);
