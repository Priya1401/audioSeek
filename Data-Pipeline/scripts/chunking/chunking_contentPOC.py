def load_transcribed_content(filename):
    with open(filename, 'r') as file:
        content = file.read()
    return content


# ============================================================================
# STEP 1: PARSE RAW TRANSCRIPT
# ============================================================================

import re

def parse_transcript(content):
  """
    Convert raw Whisper transcript into structured segments.

    Input : Raw strings with timestamps and transcriptions.
    Output: List of dictionaries with structured segments.
  """

  segments = []

  lines = content.strip().split('\n')

  for line in lines:
    line = line.strip()

    if not line:
      continue

    match = re.match(r'\[([\d.]+)-([\d.]+)]\s*(.+)', line)

    if not match:
      continue

    start_time = match.group(1)
    end_time = match.group(2)
    text = match.group(3)

    segment = {
      'segment_id' : len(segments),
      'start_time': start_time,
      'end_time': end_time,
      'duration': round(float(end_time) - float(start_time), 2),
      'text': text
    }

    segments.append(segment)


  return segments

# ============================================================================
# STEP 2: ADD FORMATED TIMESTAMPS
# ============================================================================

from datetime import timedelta

def format_timestamps(seconds):
  '''
  Convert the seconds timestamp into HH:MM:SS format for better identificationo fquestions based on time stamps
  '''
  td = timedelta(seconds=seconds)


  hours, remainder = divmod(round(td.total_seconds()), 3600)
  minutes, seconds = divmod(remainder, 60)

  if hours > 0 :
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'
  else:
    return f'{minutes:02d}:{seconds:02d}'



def update_timestamps(segments):
  '''
  Add formatted timestamps to each segment.
  '''
  for segment in segments:
    segment['formatted_start'] = format_timestamps(float(segment['start_time']))
    segment['formatted_end'] = format_timestamps(float(segment['end_time']))

  return segments


def process_transcripts(filename):
    raw_transcript = load_transcribed_content(filename)
    segments = parse_transcript(raw_transcript)
    segments = update_timestamps(segments)

    print("\nPHASE 1 COMPLETE\n")
    print("SUMMARY STATISTICS")

    total_duration = float(segments[-1]['end_time']) - float(segments[0]['start_time'])
    print(f"Total segments: {len(segments)}")
    print(f"Total duration: {format_timestamps(total_duration)} ({total_duration:.1f} seconds)")
    print(f"Avg segment length: {total_duration / len(segments):.1f} seconds")
    print(f"First segment starts: {segments[0]['formatted_start']}")
    print(f"Last segment ends: {segments[-1]['formatted_end']}")
    return segments

chapter_segments = process_transcripts("data/transcription_results/faster_whisper.txt")

# ============================================================================
# QUICK ACCESS FUNCTIONS
# ============================================================================

def get_segment_at_time(segments, time_seconds):
    """Find which segment contains a specific timestamp"""
    for seg in segments:
        if seg['start_time'] <= time_seconds <= seg['end_time']:
            return seg
    return None


def get_segments_in_range(segments, start_time, end_time):
    """Get all segments within a time range"""
    return [
        seg for seg in segments
        if seg['start_time'] <= end_time and seg['end_time'] >= start_time
    ]


def search_text(segments, query):
    """Simple keyword search in segments"""
    query_lower = query.lower()
    return [
        seg for seg in segments
        if query_lower in seg['text'].lower()
    ]


# Based on english lanmguage each token represenst about 0.75 word (3/4 word)


def count_token(text):
    words = text.split()
    return len(words) * 1.33


def create_chunk_dict(segments, chunk_id):
    combined_text = ' '.join([seg['text'] for seg in segments])

    start_time = segments[0]['start_time']
    end_time = segments[-1]['end_time']
    duration = float(segments[-1]['end_time']) - float(segments[0]['start_time'])

    formatted_start_time = segments[0]['formatted_start']
    formatted_end_time = segments[-1]['formatted_end']

    chunk = {
        'chunk_id': chunk_id,
        'start_time': start_time,
        'end_time': end_time,
        'duration': round(duration, 2),
        'formatted_start_time': formatted_start_time,
        'formatted_end_time': formatted_end_time,
        'token_count': count_token(combined_text),
        'segment_count': len(segments),
        'segment_ids': [seg['segment_id'] for seg in segments],
        'segments': segments
    }

    return chunk


def create_chunks(segments, target_tokens=512, overlap_tokens=100, warn_oversized=True):
    print("inside create chunks")
    oversized_segments = []
    max_segment_tokens = 0

    chunks = []
    current_chunk_segments = []
    current_tokens = 0
    chunk_id = 0

    i = 0
    while i < len(segments):
        print(f"Checking segment {i}")
        segment = segments[i]
        segment_tokens = count_token(segment['text'])

        if segment_tokens > max_segment_tokens:
            max_segment_tokens = segment_tokens

        if segment_tokens > target_tokens:
            oversized_segments.append({
                'segment_id': segment.get('segment_id', i),
                'tokens': segment_tokens,
                'excess': segment_tokens - target_tokens,
                'excess_pct': (segment_tokens - target_tokens) / target_tokens * 100,
                'time': segment.get('formatted_start', 'Unknown')
            })

        if current_tokens + segment_tokens > target_tokens and current_chunk_segments:
            chunk = create_chunk_dict(current_chunk_segments, chunk_id)
            print(f"Added chunk {chunk_id}")
            chunks.append(chunk)
            chunk_id += 1

            overlap_segments = []
            overlap_token_count = 0

            for seg in reversed(current_chunk_segments):
                seg_tokens = count_token(seg['text'])
                if overlap_token_count + seg_tokens <= overlap_tokens:
                    overlap_segments.insert(0, seg)
                    overlap_token_count += seg_tokens
                else:
                    break

            current_chunk_segments = overlap_segments
            current_tokens = overlap_token_count

        current_chunk_segments.append(segment)
        current_tokens += segment_tokens
        i += 1

    if current_chunk_segments:
        chunk = create_chunk_dict(current_chunk_segments, chunk_id)
        chunks.append(chunk)

    print(f"Created {len(chunks)} chunks")
    print(f"Avg chunk size: {sum(c['token_count'] for c in chunks) / len(chunks):.0f} tokens")
    print(f"Avg chunk duration: {sum(c['duration'] for c in chunks) / len(chunks):.1f} seconds")

    if oversized_segments and warn_oversized:
        print(f"\n Found {len(oversized_segments)} oversized segments:")
        print(f"Largest segment: {max_segment_tokens} tokens (target: {target_tokens})")
        print(f"These segments are standalone chunks")
        if max_segment_tokens > target_tokens * 1.5:
            recommended_size = int(max_segment_tokens * 1.2)
            print(f"Recommended chunk size: {recommended_size} tokens")
        else:
            print(f"Acceptable oversized segments can contiunue with {target_tokens} tokens")

    return chunks

# ============================================================================
# QUICK ACCESS FUNCTIONS
# ============================================================================

def filter_chunks_prior_timestamp(chunks, current_position):
  filtered = [ chunk for chunk in chunks if float(chunk['start_time']) <= current_position]
  return filtered

def get_chunks_in_time_range(chunks, start_time, end_time):
  return [
    chunk for chunk in chunks
    if float(chunk['start_time']) <= end_time and float(chunk['end_time']) >= start_time
  ]

def retrieve_chunks_with_keyword(chunks, keyword):
  keyword_lower = keyword.lower()
  return [
    chunk for chunk in chunks
    if any(keyword_lower in seg['text'].lower() for seg in chunk['segments'])
  ]


import json
def save_chunks(chunks, filepath = 'chunks.json'):
    """
    Save chunks to JSON file
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"\n Saved {len(chunks)} chunks to {filepath}")


def load_chunks(filepath: str = 'chunks.json') :
    """
    Load chunks from JSON file
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks from {filepath}")
    return chunks


def process_chunks(segments, target_tokens = 512, overlap_tokens = 100, save_file = 'data/chunking_results/chunks.json'):
  print("in Main call")
  chunks = create_chunks(segments, target_tokens, overlap_tokens)
  save_chunks(chunks, save_file)
  return chunks

chunks = process_chunks(chapter_segments)

filtered = get_chunks_in_time_range(chunks, 1000, 1500)
print("Chunks from time range : " )
for i in range(len(filtered)):
  print(filtered[i]['chunk_id'])

keyword = "Wednesday"
filtered = retrieve_chunks_with_keyword(chunks, keyword)
print("Keyword Search : " )
for i in range(len(filtered)):
  print(filtered[i]['chunk_id'])