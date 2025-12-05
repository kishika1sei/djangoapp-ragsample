# RAGチャットボットアプリ フロー定義

## ユーザ質問フロー

1. User がチャット画面からクエリ送信
2. ChatSession / ChatMessage を保存
3. EmbeddingService がクエリをベクトル化
4. SearchBackend.search() を呼び出し、トップKチャンク取得
5. PromptBuilder が
   1. システムメッセージ
   2. コンテキスト（取得チャンク）
   3. ユーザーメッセージ
    を組み立て
6. LLM API コール
7. 応答を ChatMessage（role=assistant）として保存
8. レスポンスをフロントに返却

## ドキュメントアップロード～インデックス更新フロー

1. User が管理画面からファイルアップロード
2. Document レコード作成（department, uploaded_by, file_path 等）
3. IngestionService が
   1. PDF/TXT/CSV を読み込み
   2. ページ単位テキスト抽出
   3. チャンク分割（chunk_index, page）
4. 各チャンクごとに
   1. embedding 生成
   2. Chunk レコード作成
   3. SearchBackend.index_chunks() でインデックスに反映
5. 完了ステータスをDocumentに記録（必要なら）
