#!/usr/bin/env python3
"""
F2 Discord AI Bot
- Responds when pinged or mentioned
- Uses custom embedding + synthesis engine
- No external API calls required
- Auto clears cache on startup for fresh data
- Optimized with dynamic sentence selection based on relevance thresholds for concise answers
- Limited to first four sentences in response
"""


import discord
from discord.ext import commands
import os
import re
import asyncio
import numpy as np
from pathlib import Path
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import requests
import aiohttp
import json
import datetime
import hashlib
import chromadb
import spacy
from readability import Document
from dotenv import load_dotenv
import warnings


warnings.filterwarnings('ignore')
load_dotenv()


# ---------------------------
# CONFIG
# ---------------------------
CONFIG = {
   'max_context_length': 8000,
   'search_timeout': 0,
   'max_sources': 10000,
   'response_quality': 'high',
   'min_sim_threshold': 1,
   'min_relevance_threshold': 1,
}


TOKEN = "INSERT_BOT_TOKEN"
INTENTS = discord.Intents.default()
INTENTS.message_content = True
INTENTS.guilds = True
INTENTS.members = True


# Load SpaCy model
try:
   nlp = spacy.load("en_core_web_sm")
except OSError:
   print("SpaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
   exit(1)


# ---------------------------
# MEMORY SYSTEM
# ---------------------------
class AdvancedMemory:
   def __init__(self, db_path="./chroma_db"):
       self.conversations = {}
       self.client = chromadb.PersistentClient(path=db_path)
       # Auto clear cache on startup
       try:
           self.client.delete_collection(name="knowledge")
           print("Knowledge base cache cleared on startup.")
       except Exception:
           pass
       self.knowledge_base = self.client.get_or_create_collection(name="knowledge")
       self.knowledge_graph = {'entities': {}}


   def add_message(self, user_id: str, role: str, content: str):
       if user_id not in self.conversations:
           self.conversations[user_id] = []
       self.conversations[user_id].append({
           'role': role,
           'content': content,
           'timestamp': datetime.datetime.now().isoformat()
       })
       # Limit to last 1 for no history carry-over
       if len(self.conversations[user_id]) > 1:
           self.conversations[user_id] = self.conversations[user_id][-1:]


   def clear_user_history(self, user_id: str):
       if user_id in self.conversations:
           del self.conversations[user_id]


MEMORY = AdvancedMemory()
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')


# ---------------------------
# WEB CRAWLER
# ---------------------------
class AdvancedWebCrawler:
   def __init__(self):
       self.session = None


   async def init_session(self):
       if self.session is None or self.session.closed:
           self.session = aiohttp.ClientSession(
               headers={"User-Agent": "Mozilla/5.0", "Accept-Encoding": "gzip, deflate"},
               timeout=aiohttp.ClientTimeout(total=CONFIG['search_timeout'])
           )


   async def close_session(self):
       if self.session and not self.session.closed:
           await self.session.close()


   def _extract_knowledge(self, text, url):
       doc = nlp(text)
       for sent in doc.sents:
           sent_text = sent.text.strip()
           sent_doc = nlp(sent_text)
           for ent in sent_doc.ents:
               if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT']:
                   entity_key = ent.text.lower().strip()
                   if entity_key not in MEMORY.knowledge_graph['entities']:
                       MEMORY.knowledge_graph['entities'][entity_key] = {
                           'label': ent.label_,
                           'sources': []
                       }
                   source_info = {'sentence': sent_text, 'url': url}
                   if source_info not in MEMORY.knowledge_graph['entities'][entity_key]['sources']:
                       MEMORY.knowledge_graph['entities'][entity_key]['sources'].append(source_info)


   async def search_web(self, query):
       await self.init_session()
       query_emb = EMBEDDING_MODEL.encode([query]).tolist()
       try:
           results = MEMORY.knowledge_base.query(query_embeddings=query_emb, n_results=CONFIG['max_sources'])
       except Exception:
           results = None


       sources_from_db = []
       if results and results.get('documents'):
           for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
               if dist < 0.7:
                   sources_from_db.append({
                       'source': meta['source'],
                       'content': doc,
                       'url': meta['url'],
                       'quality_score': meta.get('quality_score', 0.8)
                   })


       if len(sources_from_db) < CONFIG['max_sources']:
           new_sources = await self._fetch_general_web(query)
           self._store_sources(new_sources)
           all_sources = self._deduplicate_sources(sources_from_db + new_sources)
       else:
           all_sources = sources_from_db


       ranked = self._rank_sources(query, all_sources)
       return ranked[:CONFIG['max_sources']]


   async def _fetch_general_web(self, query):
       try:
           from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
           search = DuckDuckGoSearchAPIWrapper(max_results=CONFIG['max_sources'] // 2)
           results = search.results(query, max_results=CONFIG['max_sources'] // 2)
           content_tasks = [self._fetch_content(r['link']) for r in results]
           contents = await asyncio.gather(*content_tasks)
           return [
               {'source': r['title'], 'content': c, 'url': r['link'], 'quality_score': 0.8}
               for r, c in zip(results, contents) if c
           ]
       except Exception as e:
           print(f"Search error: {e}")
       return []


   async def _fetch_content(self, url):
       try:
           async with self.session.get(url) as resp:
               if resp.status == 200:
                   html = await resp.text()
                   doc = Document(html)
                   soup = BeautifulSoup(doc.summary(), 'html.parser')
                   return re.sub(r'\s+', ' ', soup.get_text(strip=True))
       except Exception:
           pass
       return ""


   def _store_sources(self, sources):
       if not sources:
           return
       docs, metas, ids = [], [], []
       for s in sources:
           if s.get('content'):
               docs.append(s['content'])
               metas.append({'source': s['source'], 'url': s['url'], 'quality_score': s['quality_score']})
               ids.append(hashlib.md5(s['content'].encode()).hexdigest())
               self._extract_knowledge(s['content'], s['url'])
       try:
           embeddings = EMBEDDING_MODEL.encode(docs).tolist()
           MEMORY.knowledge_base.add(documents=docs, embeddings=embeddings, metadatas=metas, ids=ids)
       except Exception as e:
           print(f"ChromaDB add failed: {e}")


   def _deduplicate_sources(self, sources):
       seen = set()
       unique = []
       for s in sources:
           if s and s.get('url') not in seen:
               seen.add(s['url'])
               unique.append(s)
       return unique


   def _rank_sources(self, query, sources):
       if not sources:
           return []
       q_emb = EMBEDDING_MODEL.encode([query])[0]
       for s in sources:
           if s.get('content'):
               c_emb = EMBEDDING_MODEL.encode([s['content']])[0]
               s['relevance_score'] = np.dot(q_emb, c_emb) * s.get('quality_score', 0.8)
       return sorted(sources, key=lambda x: x.get('relevance_score', 0), reverse=True)


# ---------------------------
# SEMANTIC SYNTHESIS ENGINE
# ---------------------------
class SynthesisEngine:
   def generate_response(self, query, sources):
       kg = self._query_knowledge_graph(query)
       if kg:
           return {'main_response': kg['response'], 'citations': [kg['citation']], 'sources_used': 1, 'confidence_score': 0.95}


       if not sources:
           return self._fallback(query)


       text = " ".join(s['content'] for s in sources if s.get('content'))
       sentences = [s.text.strip() for s in nlp(text).sents if self._valid_sentence(s.text)]
       if not sentences:
           return self._fallback(query)


       query_vec = EMBEDDING_MODEL.encode([query])[0]
       sent_vecs = EMBEDDING_MODEL.encode(sentences)
       sims = np.dot(sent_vecs, query_vec)
       # Dynamic selection: only sentences above min threshold
       ranked = [(s, float(sims[i])) for i, s in enumerate(sentences) if sims[i] > CONFIG['min_sim_threshold']]
       if not ranked:
           return self._fallback(query)
       ranked.sort(key=lambda x: x[1], reverse=True)
       top = [s for s, _ in ranked]


       response = self._semantic_summarize(query, top, sims)
       citations = self._citations(sources)
       avg_sim = np.mean([sim for sim in sims if sim > CONFIG['min_sim_threshold']]) if ranked else 0
       return {'main_response': response, 'citations': citations, 'sources_used': len(sources), 'confidence_score': min(1.0, avg_sim + 0.3)}


   def _valid_sentence(self, s):
       s = s.strip()
       if not s or len(s.split()) < 4 or s.endswith('?'):
           return False
       doc = nlp(s, disable=["parser", "ner"])
       return any(t.pos_ == 'VERB' for t in doc)


   def _semantic_summarize(self, query, sentences, sims):
       embeddings = EMBEDDING_MODEL.encode(sentences)
       centroid = np.mean(embeddings, axis=0)
       relevance = np.dot(embeddings, centroid)
       # Dynamic: select only above relevance threshold
       ordered_with_rel = [(s, rel) for s, rel in zip(sentences, relevance) if rel > CONFIG['min_relevance_threshold']]
       if not ordered_with_rel:
           ordered = sentences[:3]  # Fallback to top 3 if none meet threshold
       else:
           ordered_with_rel.sort(key=lambda x: x[1], reverse=True)
           ordered = [s for s, _ in ordered_with_rel[:3]]  # Limit to top 3 for conciseness
       key_points = []
       for s in ordered:
           clean = self._clean_sentence(s)
           if clean and all(clean not in kp for kp in key_points):
               key_points.append(clean)
       opener = f"{query.capitalize()} — here's a concise overview:" if len(query) > 3 else "Here's a concise overview:"
       body = " ".join(self._link_sentences(key_points))
       return f"{opener}\n\n{self._smooth_text(body)}"


   def _clean_sentence(self, s):
       s = re.sub(r'\s+', ' ', s).strip()
       s = re.sub(r'^(but|and|so|however|then|on the other hand)[, ]+', '', s, flags=re.I)
       if s and not s[0].isupper():
           s = s[0].upper() + s[1:]
       return s


   def _link_sentences(self, sentences):
       transitions = ["Additionally,", "Furthermore,", "Interestingly,", "In essence,", "Overall,", "Moreover,", "Notably,", "In addition,"]
       merged = [sentences[0]] if sentences else []
       for i, s in enumerate(sentences[1:], 1):
           if len(s.split()) >= 5:
               merged.append(f"{transitions[i % len(transitions)]} {s}")
       return merged


   def _smooth_text(self, text):
       text = re.sub(r'(\s)([.,])', r'\2', text)
       text = re.sub(r'\s+', ' ', text)
       return text.strip()


   def _query_knowledge_graph(self, query):
       doc = nlp(query.lower())
       if "who is" in query.lower() or "what is" in query.lower():
           ents = [e.text.lower() for e in doc.ents]
           if ents:
               e = ents[0]
               if e in MEMORY.knowledge_graph['entities']:
                   src = MEMORY.knowledge_graph['entities'][e]['sources'][0]
                   return {'response': f"From memory, {e.title()} is known as: {src['sentence']}", 'citation': f"1. {src['url']}"}
       return None


   def _citations(self, sources):
       return [f"{i}. {s.get('source','Web')} - <{s.get('url','')}>" for i, s in enumerate(sources, 1)]


   def _fallback(self, query):
       return {'main_response': f"I couldn't find detailed info on **{query}**, but try rephrasing for better results.",
               'citations': [], 'sources_used': 0, 'confidence_score': 0.1}


# ---------------------------
# F2 CORE
# ---------------------------
class F2_AI:
   def __init__(self):
       self.crawler = AdvancedWebCrawler()
       self.generator = SynthesisEngine()


   async def ask(self, query, user_id):
       sources = await self.crawler.search_web(query)
       response = self.generator.generate_response(query, sources)
       MEMORY.add_message(user_id, 'user', query)
       MEMORY.add_message(user_id, 'assistant', response['main_response'])
       # Auto clear user history after response for no carry-over
       MEMORY.clear_user_history(user_id)
       return response


# ---------------------------
# DISCORD BOT SETUP
# ---------------------------


bot = commands.Bot(command_prefix="!", intents=INTENTS)
ai = F2_AI()


@bot.event
async def on_ready():
   print(f"✅ Logged in as {bot.user} (ID: {bot.user.id})")
   await bot.change_presence(activity=discord.Activity(type=discord.ActivityType.listening, name="you"))


@bot.event
async def on_message(message: discord.Message):
   if message.author.bot:
       return


   # Respond only if bot is pinged or mentioned
   if bot.user in message.mentions:
       query = message.content.replace(f"<@!{bot.user.id}>", "").replace(f"<@{bot.user.id}>", "").strip()
       if not query:
           await message.reply("👋 Hey! Ask me something — just mention me with your question.")
           return


       async with message.channel.typing():
           result = await ai.ask(query, str(message.author.id))
           response = result['main_response']
           if result.get('citations'):
               response += "\n\n📚 Sources:\n" + "\n".join(result['citations'])
           response += f"\n\nConfidence: {result.get('confidence_score', 0.0):.0%}"


       await message.reply(response[:1900])  # Discord message limit safeguard


# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
   bot.run(TOKEN)
