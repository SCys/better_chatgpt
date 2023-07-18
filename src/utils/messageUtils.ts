import { MessageInterface, ModelOptions, TotalTokenUsed } from '@type/chat';

import useStore from '@store/store';

import { Tiktoken } from '@dqbd/tiktoken/lite';
const cl100k_base = await import('@dqbd/tiktoken/encoders/cl100k_base.json');

const encoder = new Tiktoken(
  cl100k_base.bpe_ranks,
  {
    ...cl100k_base.special_tokens,
    '<|im_start|>': 100264,
    '<|im_end|>': 100265,
    '<|im_sep|>': 100266,
  },
  cl100k_base.pat_str
);

// https://github.com/dqbd/tiktoken/issues/23#issuecomment-1483317174
export const getChatGPTEncoding = (
  messages: MessageInterface[],
  model: ModelOptions
) => {
  const isGpt3 = model === 'gpt-3.5-turbo';

  const msgSep = isGpt3 ? '\n' : '';
  const roleSep = isGpt3 ? '\n' : '<|im_sep|>';

  const serialized = [
    messages
      .map(({ role, content }) => {
        const _content = toHalfWidth(content);
        return `<|im_start|>${role}${roleSep}${_content}<|im_end|>`;
      })
      .join(msgSep),
    `<|im_start|>assistant${roleSep}`,
  ].join(msgSep);

  return encoder.encode(serialized, 'all');
};

const countTokens = (messages: MessageInterface[], model: ModelOptions) => {
  if (messages.length === 0) return 0;
  return getChatGPTEncoding(messages, model).length;
};

export const limitMessageTokens = (
  messages: MessageInterface[],
  limit: number = 4096,
  model: ModelOptions
): MessageInterface[] => {
  const limitedMessages: MessageInterface[] = [];
  let tokenCount = 0;

  const isSystemFirstMessage = messages[0]?.role === 'system';
  let retainSystemMessage = false;

  // Check if the first message is a system message and if it fits within the token limit
  if (isSystemFirstMessage) {
    const systemTokenCount = countTokens([messages[0]], model);
    if (systemTokenCount < limit) {
      tokenCount += systemTokenCount;
      retainSystemMessage = true;
    }
  }

  // Iterate through messages in reverse order, adding them to the limitedMessages array
  // until the token limit is reached (excludes first message)
  for (let i = messages.length - 1; i >= 1; i--) {
    const count = countTokens([messages[i]], model);
    if (count + tokenCount > limit) break;
    tokenCount += count;
    limitedMessages.unshift({ ...messages[i] });
  }

  // Process first message
  if (retainSystemMessage) {
    // Insert the system message in the third position from the end
    limitedMessages.splice(-3, 0, { ...messages[0] });
  } else if (!isSystemFirstMessage) {
    // Check if the first message (non-system) can fit within the limit
    const firstMessageTokenCount = countTokens([messages[0]], model);
    if (firstMessageTokenCount + tokenCount < limit) {
      limitedMessages.unshift({ ...messages[0] });
    }
  }

  // convert to half width
  limitedMessages.forEach((message) => {
    message.content = toHalfWidth(message.content);
  });

  return limitedMessages;
};

export const updateTotalTokenUsed = (
  model: ModelOptions,
  promptMessages: MessageInterface[],
  completionMessage: MessageInterface
) => {
  const setTotalTokenUsed = useStore.getState().setTotalTokenUsed;
  const updatedTotalTokenUsed: TotalTokenUsed = JSON.parse(
    JSON.stringify(useStore.getState().totalTokenUsed)
  );

  const newPromptTokens = countTokens(promptMessages, model);
  const newCompletionTokens = countTokens([completionMessage], model);
  const { promptTokens = 0, completionTokens = 0 } =
    updatedTotalTokenUsed[model] ?? {};

  updatedTotalTokenUsed[model] = {
    promptTokens: promptTokens + newPromptTokens,
    completionTokens: completionTokens + newCompletionTokens,
  };
  setTotalTokenUsed(updatedTotalTokenUsed);
};

export default countTokens;

const halfWidthMap: Map<string, string> = new Map([
  ['“', '"'],
  ['”', '"'],
  ['‘', "'"],
  ['’', "'"],
]);

function toHalfWidth(text: string): string {
  const fullWidthSymbolsRegex = /[Ａ-Ｚａ-ｚ０-９（）［］｛｝＜＞＆＃＠＊＋－＝～！％＇＂，．／：；？＼＾＿｀｜￣]/g;
  const halfWidthSymbolsMapping: { [key: string]: string } = {
    'Ａ': 'A', 'Ｂ': 'B', 'Ｃ': 'C', 'Ｄ': 'D', 'Ｅ': 'E', 'Ｆ': 'F', 'Ｇ': 'G', 'Ｈ': 'H', 'Ｉ': 'I', 'Ｊ': 'J',
    'Ｋ': 'K', 'Ｌ': 'L', 'Ｍ': 'M', 'Ｎ': 'N', 'Ｏ': 'O', 'Ｐ': 'P', 'Ｑ': 'Q', 'Ｒ': 'R', 'Ｓ': 'S', 'Ｔ': 'T',
    'Ｕ': 'U', 'Ｖ': 'V', 'Ｗ': 'W', 'Ｘ': 'X', 'Ｙ': 'Y', 'Ｚ': 'Z', 'ａ': 'a', 'ｂ': 'b', 'ｃ': 'c', 'ｄ': 'd',
    'ｅ': 'e', 'ｆ': 'f', 'ｇ': 'g', 'ｈ': 'h', 'ｉ': 'i', 'ｊ': 'j', 'ｋ': 'k', 'ｌ': 'l', 'ｍ': 'm', 'ｎ': 'n',
    'ｏ': 'o', 'ｐ': 'p', 'ｑ': 'q', 'ｒ': 'r', 'ｓ': 's', 'ｔ': 't', 'ｕ': 'u', 'ｖ': 'v', 'ｗ': 'w', 'ｘ': 'x',
    'ｙ': 'y', 'ｚ': 'z', '０': '0', '１': '1', '２': '2', '３': '3', '４': '4', '５': '5', '６': '6', '７': '7',
    '８': '8', '９': '9', '（': '(', '）': ')', '［': '[', '］': ']', '｛': '{', '｝': '}', '＜': '<', '＞': '>',
    '＆': '&', '＃': '#', '＠': '@', '＊': '*', '＋': '+', '－': '-', '＝': '=', '～': '~', '！': '!', '％': '%',
    '＇': "'", '＂': '"', '，': ',', '．': '.', '／': '/', '：': ':', '；': ';', '？': '?', '＼': '\\', '＾': '^',
    '＿': '_', '｀': '`', '｜': '|', '￣': '~'
  };

  return text.replace(fullWidthSymbolsRegex, (match) => {
    return halfWidthSymbolsMapping[match] || match;
  });
}