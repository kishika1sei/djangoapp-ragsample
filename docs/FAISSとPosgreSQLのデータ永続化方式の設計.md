# FAISSとPosgreSQLのデータ永続化設計

## お断り事項

- アプリ開発着手前に、データの扱いについて基本方針を定めたものです。
- 下記の基本方針、保存場所、データライフサイクル設計に基づいて、開発をしていますが、詳細部分が異なる場合がございます。

## 方針

- PostgreSQLはデータのソース元として保管(Chunkテーブル + embedding列)
- FAISS: 起動時 or バッチ時にDBから再構築できる"インデックスキャッシュ"

## 保存場所

- Chunkembedding : pgvectorまたはJSON / ArrayField
- FAISS index:アプリサーバーのローカルディスクに.indexファイルとして保存

## データライフサイクル設計

1. 初回構築
   1. 管理コマンドrebuild_faiss_indexを作成
   2. Chunk全件をDBから読み込み、FAISS indexを構築、ファイルに保存
2. インクリメンタル更新
   1. ドキュメントアップロード時の最後に
      1. 新規ChunkのembeddingをFAISSに add_with_ids
      2. indexファイルを保存(一定件数ごと、or バッチ)
3. 再構築
   1. 管理画面に「インデックス再構築」ボタン
   2. 押されたらSearchBackend.rebuild_index()を実行
4. 復旧
   1. アプリ起動時
      1. indexファイルがあればロード
      2. なければrebuild_index()を実行
