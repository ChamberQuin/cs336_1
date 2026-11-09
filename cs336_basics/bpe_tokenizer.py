import cProfile
from concurrent.futures import ProcessPoolExecutor, as_completed
import pstats
import regex as re
from regex import Pattern

from collections import defaultdict
import time
from typing import Iterator

from cs336_basics.pretokenization_example import find_chunk_boundaries

Pair = tuple[bytes, bytes]
Sequence = tuple[bytes, ...]

# pre-tokenize regex pat, same across trainers
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
ENCODING = "utf-8"

class BPETokenizerTrainer:
    special_tokens: list[str] = ["<|endoftext|>"]
    # final vocab size, including special tokens, initial bytes and merges
    vocab_size: int
    # token id -> token bytes
    vocab: dict[int, bytes]
    merges: list[Pair]

    def __init__(self, vocab_size, special_tokens):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

        # 预编译 regex
        self.pre_token_regex = re.compile(PAT)
        # 先用 re.escape 转义 special_token 中的特殊字符，再用 "|" 拼接正则，避免正则被特殊字符搞坏
        self.split_regex = re.compile(
            "|".join(map(re.escape, special_tokens))
        )

        # 初始化 vocab, including special tokens, initial bytes
        ## bytes([65]) 是 b'A', 而 bytes(65) 是创建 65 个零字节
        self.vocab = {i: bytes([i]) for i in range(256)}
        for idx, special_token in enumerate(special_tokens):
            self.vocab[256 + idx] = special_token.encode(ENCODING)
        
        self.merges = []

    def naive_train(self, text: str):
        # i.e. {(l,o,w):5, (l,o,w,e,r):2, (w,i,d,e,s,t): 3, (n,e,w,e,s,t): 6}
        pre_token_freq : dict[Sequence, int] = defaultdict(int)

        # pre-tokenize, split text by special token, handle marks and represent with byte sequence
        # input string -> pre-token -> byte sequence
        chunks = self.split_regex.split(text)
        for chunk in chunks:
            # use finditer instead of findall to avoid copying large data
            for m in self.pre_token_regex.finditer(chunk):
                pre_token_bytes = m.group(0).encode(ENCODING)
                pre_token_seq = tuple([pre_token.to_bytes() for pre_token in pre_token_bytes])
                # print(pre_token_seq)
                pre_token_freq[pre_token_seq] += 1

        start_time = time.time()

        merge_cnt = self.vocab_size - len(self.vocab)
        for epoch in range(merge_cnt):
            # print(f"epoch: {epoch}")
            byte_pair_freq : dict[Pair, int] = defaultdict(int)

            for pre_token, freq in pre_token_freq.items():
                # print(f"pre_token: {pre_token}, freq: {freq}")
                for i in range(len(pre_token) - 1):
                    byte_pair_freq[(pre_token[i], pre_token[i + 1])] += freq
            # print(f"pre_token_freq: {len(pre_token_freq)}, byte_pair_freq: {len(byte_pair_freq)}, vocab: {len(self.vocab)}")

            if len(byte_pair_freq) == 0:
                break

            candidate_pair = ()
            candidate_freq = 0
            for pair, freq in byte_pair_freq.items():
                if freq > candidate_freq:
                    candidate_pair = pair
                    candidate_freq = freq
                elif freq == candidate_freq and pair > candidate_pair:
                    candidate_pair = pair

            self.vocab[len(self.vocab)] = candidate_pair[0] + candidate_pair[1]
            self.merges.append((candidate_pair[0], candidate_pair[1]))

            # re-cal pre_token frequency
            new_pre_token_freq = {}
            for pre_token, freq in pre_token_freq.items():
                idx = 0
                new_pre_token_seq = []
                while idx < len(pre_token):
                    if (idx < len(pre_token) - 1) and (pre_token[idx], pre_token[idx + 1]) == candidate_pair:
                        new_pre_token_seq.append(pre_token[idx] + pre_token[idx + 1])
                        idx += 2
                    else:
                        new_pre_token_seq.append(pre_token[idx])
                        idx += 1
                new_pre_token_freq[tuple(new_pre_token_seq)] = freq

            pre_token_freq = new_pre_token_freq

        print(f"merge {merge_cnt} times, cost {time.time() - start_time}")
        print(f"merges: {self.merges}")

    # opt1: 
    # threading 适合 IO 密集，但 CPU 密集没明显加速，因为有 GIL（Global Interpreter Lock，同一时刻通常只有一个线程能执行 Python 字节码）,线程等待 IO 时会释放 GIL；
    # multiprocessing 适合 CPU 密集。
    # 也可以用 NumPy/PyTorch 这类底层释放 GIL 的库，调用 CPP/Rust 扩展，或用多进程 multiprocessing 绕过 GIL 从而实现并行计算。
    def train_from_scratch(
        self,
        path: str,
        process_num: int,
        chunk_num: int = 16,
    ):
        # read chunks from path
        start_time = time.time()
        chunk_iter = BPETokenizerTrainer.read_from_path(path, chunk_num)
        chunks = []
        for chunk in chunk_iter:
            chunks.append(chunk)
        print(f"read io: {time.time() - start_time}")

        # batch pre-tokenize
        start_time = time.time()
        pre_token_freq = self.batch_pre_tokenize(chunks, process_num)
        print(f"pre-tokenize: {time.time() - start_time}")

        # train
        start_time = time.time()
        
        # profiler = cProfile.Profile()
        # profiler.enable()
        
        self.train(pre_token_freq)

        # profiler.disable()
        # stats = pstats.Stats(profiler)
        # stats.sort_stats("cumulative").print_stats(10)
        
        print(f"train: {time.time() - start_time}")

    @staticmethod
    def read_from_path(path: str, chunk_num: int) -> Iterator[str]:
        with open(path, "rb") as f:
            boundaries = find_chunk_boundaries(f, chunk_num, b"<|endoftext|>")

            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                # Run pre-tokenization on your chunk and store the counts for each pre-token
                yield chunk

    def batch_pre_tokenize(self, chunks: list[str], process_num: int) -> dict[Sequence, int]:
        merged : dict[Sequence, int] = defaultdict(int)
        # 用进程不用线程，避免 GIL
        with ProcessPoolExecutor(process_num) as executor:
            futures = [
                executor.submit(
                    BPETokenizerTrainer.pre_tokenize, chunk, self.split_regex, self.pre_token_regex
                ) for chunk in chunks
            ]
            for future in as_completed(futures):
                for pre_token_seq, freq in future.result().items():
                    merged[pre_token_seq] += freq
        return merged

    @staticmethod
    def pre_tokenize(chunk: str, split_regex: Pattern[str], pre_token_regex: Pattern[str]) -> dict[Sequence, int]:
        pre_token_freq : dict[Sequence, int] = defaultdict(int)
        for seg in split_regex.split(chunk):
            for word in pre_token_regex.finditer(seg):
                pre_token_bytes = word.group(0).encode(ENCODING)
                pre_token_seq = tuple([pre_token.to_bytes() for pre_token in pre_token_bytes])
                pre_token_freq[pre_token_seq] += 1
        return pre_token_freq

    def train(self, pre_token_freq: dict[Sequence, int]):
        seq_by_id, seq_id_freq, pair_freq, pair_in_seq = BPETokenizerTrainer.build_indexes(pre_token_freq)

        assert len(self.merges) == 0
        merge_cnt = self.vocab_size - len(self.vocab)
        for epoch in range(merge_cnt):
            if not pair_freq:
                # all pairs has been merged
                break

            # todo: max heap
            best_pair = max(pair_freq, key = lambda p: (pair_freq[p], p))
            
            self.vocab[len(self.vocab)] = best_pair[0] + best_pair[1]
            self.merges.append(best_pair)
            # print(f"epoch:{epoch}, best_pair:{best_pair}, vocab:{self.vocab}, merges:{self.merges}")

            affected_ids = list(pair_in_seq[best_pair].keys())
            for seq_id in affected_ids:
                freq = seq_id_freq[seq_id]
                old_seq = seq_by_id[seq_id]

                new_seq, removed_pairs, added_pairs = BPETokenizerTrainer.merge_sequence(old_seq, best_pair)
                seq_by_id[seq_id] = new_seq

                for pair in removed_pairs:
                    pair_in_seq[pair][seq_id] -= 1
                    if pair_in_seq[pair][seq_id] == 0:
                        del pair_in_seq[pair][seq_id]
                        if not pair_in_seq[pair]:
                            del pair_in_seq[pair]
                    pair_freq[pair] -= freq
                    if pair_freq[pair] == 0:
                        del pair_freq[pair]

                for pair in added_pairs:
                    if pair not in pair_in_seq:
                        pair_in_seq[pair] = {}
                    pair_in_seq[pair][seq_id] = pair_in_seq[pair].get(seq_id, 0) + 1
                    pair_freq[pair] = pair_freq.get(pair, 0) + freq

    @staticmethod
    def merge_sequence(seq: Sequence, target: Pair) -> tuple[
        Sequence, # new_seq
        list[Pair], # removed_pairs
        list[Pair], # added_pairs
    ]:
        # print(f"merge_seq:{seq}, target:{target}")
        merged = target[0] + target[1]

        # generate merged sequence
        new_seq_list: list[bytes] = []
        seq_len = len(seq)
        i = 0
        while i < seq_len:
            if i < seq_len - 1 and (seq[i], seq[i + 1]) == target:
                new_seq_list.append(merged)
                i += 2
            else:
                new_seq_list.append(seq[i])
                i += 1
        new_seq = tuple(new_seq_list)
        new_seq_len = len(new_seq)

        # calculate removed and added pair count using pair count changes to handle continuous merge.
        # i.e. BC,BC
        old_pair_cnt: dict[Pair, int] = defaultdict(int)
        for i in range(seq_len - 1):
            pair = (seq[i], seq[i + 1])
            old_pair_cnt[pair] += 1
        new_pair_cnt: dict[Pair, int] = defaultdict(int)
        for i in range(new_seq_len - 1):
            pair = (new_seq[i], new_seq[i + 1])
            new_pair_cnt[pair] += 1
        removed_pairs: list[Pair] = []
        for pair, old_cnt in old_pair_cnt.items():
            new_cnt = new_pair_cnt[pair]
            removed_cnt = old_cnt - new_cnt
            if removed_cnt > 0:
                removed_pairs.extend([pair] * removed_cnt)
        added_pairs: list[Pair] = []
        for pair, new_cnt in new_pair_cnt.items():
            old_cnt = old_pair_cnt[pair]
            added_cnt = new_cnt - old_cnt
            if added_cnt > 0:
                added_pairs.extend([pair] * added_cnt)

        return new_seq, removed_pairs, added_pairs

    @staticmethod
    def merge_sequence_fault(seq: Sequence, target: Pair) -> tuple[
        Sequence, # new_seq
        list[Pair], # removed_pairs
        list[Pair], # added_pairs
    ]:
        new_seq: list[bytes] = []
        removed_pairs: list[Pair] = []
        added_pairs: list[Pair] = []

        merged = target[0] + target[1]
        seq_len = len(seq)
        i = 0
        while i < seq_len:
            # 注意处理连续 merge。假设 A B C B C D，则新序列为 A BC BC D，移除 A,B B,C C,B B,C C,D，新增 A,BC BC,BC BC,D
            if i < seq_len - 1 and (seq[i], seq[i + 1]) == target:
                # 移除重叠的 pair。假设 A B C D，则移除 A,B B,C C,D
                if i > 0:
                    removed_pairs.append((seq[i - 1], seq[i]))
                removed_pairs.append(target)
                if i < seq_len - 2:
                    removed_pairs.append((seq[i + 1], seq[i + 2]))
                # 新增重叠后重组的 pair。假设 A B C D，则新增 A,BC BC,D
                if i > 0:
                    added_pairs.append((seq[i - 1], merged))
                if i < seq_len - 2:
                    added_pairs.append((merged, seq[i + 2]))
                # update seq
                new_seq.append(merged)
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        return (tuple(new_seq), removed_pairs, added_pairs)

    @staticmethod
    def build_indexes(pre_token_freq: dict[Sequence, int]) -> tuple[
        dict[int, Sequence], # seq_by_id
        dict[int, int], # seq_id_freq
        dict[Pair, int], # pair_freq
        dict[Pair, dict[int, int]], # pair_in_seq, pair -> seq_id -> pair_cnt
    ]:
        seq_by_id = {i: seq for i, seq in enumerate(pre_token_freq)}
        seq_id_freq = {i: pre_token_freq[seq] for i, seq in seq_by_id.items()}

        pair_freq: dict[Pair, int] = defaultdict(int)
        pair_in_seq: dict[Pair, dict[int, int]] = defaultdict(defaultdict)
        for seq_id, freq in seq_id_freq.items():
            seq = seq_by_id[seq_id]
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                pair_freq[pair] += freq
                if pair not in pair_in_seq:
                    pair_in_seq[pair] = defaultdict(int)
                pair_in_seq[pair][seq_id] += 1

        return seq_by_id, seq_id_freq, pair_freq, pair_in_seq

def test_naive_train():
    raw_str = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"
    trainer = BPETokenizerTrainer(300, ["<|endoftext|>"])
    trainer.naive_train(raw_str)

class BPETokenizer:


