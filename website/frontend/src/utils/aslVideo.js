const DEFAULT_SIGN_SUFFIX = "ASL sign tutorial";
const WORD_PATTERN = /[a-zA-Z']+/g;

function normalizeSignToken(sign) {
  return String(sign || "").trim().replace(/[_-]+/g, " ");
}

export function getSignVideoQuery(sign) {
  const normalized = normalizeSignToken(sign);
  return normalized ? `${normalized} ${DEFAULT_SIGN_SUFFIX}` : DEFAULT_SIGN_SUFFIX;
}

export function getSignVideoEmbedUrl(sign) {
  const slug = normalizeSignToken(sign).toLowerCase().replace(/\s+/g, "-");
  return `https://www.signasl.org/sign/${encodeURIComponent(slug)}`;
}

export function getSignVideoSearchUrl(sign) {
  const query = getSignVideoQuery(sign);
  return `https://www.youtube.com/results?search_query=${encodeURIComponent(query)}`;
}

export function extractSignKeywords(text, maxCount = 5) {
  const words = String(text || "")
    .toLowerCase()
    .match(WORD_PATTERN) || [];

  const stopwords = new Set([
    "a",
    "an",
    "the",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "do",
    "does",
    "did",
    "to",
    "of",
    "for",
    "in",
    "on",
    "at",
    "by",
    "with",
    "from",
    "as",
    "and",
    "or",
    "but",
    "if",
    "then",
    "so",
    "because",
    "that",
    "this",
    "these",
    "those",
    "can",
    "could",
    "would",
    "should",
    "will",
    "shall",
    "may",
    "might",
    "must",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
  ]);

  const keywords = [];
  for (const word of words) {
    if (stopwords.has(word)) {
      continue;
    }

    const upper = word.toUpperCase();
    if (!keywords.includes(upper)) {
      keywords.push(upper);
    }

    if (keywords.length >= maxCount) {
      break;
    }
  }

  return keywords;
}

export function buildSignChunksFromText(text, maxChunks = 5) {
  const words = String(text || "")
    .toLowerCase()
    .match(WORD_PATTERN) || [];

  const stopwords = new Set([
    "a",
    "an",
    "the",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "do",
    "does",
    "did",
    "to",
    "of",
    "for",
    "in",
    "on",
    "at",
    "by",
    "with",
    "from",
    "as",
    "and",
    "or",
    "but",
    "if",
    "then",
    "so",
    "because",
    "that",
    "this",
    "these",
    "those",
    "can",
    "could",
    "would",
    "should",
    "will",
    "shall",
    "may",
    "might",
    "must",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
  ]);

  const contentWords = words.filter((word) => !stopwords.has(word));
  if (contentWords.length === 0) {
    return [];
  }

  const chunkSize = contentWords.length > 4 ? 3 : 2;
  const chunks = [];

  for (let index = 0; index < contentWords.length && chunks.length < maxChunks; index += chunkSize) {
    chunks.push(contentWords.slice(index, index + chunkSize).join(" ").toUpperCase());
  }

  return chunks;
}

export function isPlaceholderGloss(tokens) {
  const placeholderTokens = new Set(["DUMMY", "RESPONSE", "SIGNS", "PLACEHOLDER"]);
  return Array.isArray(tokens) && tokens.length > 0 && tokens.every((token) => placeholderTokens.has(String(token || "").toUpperCase()));
}
