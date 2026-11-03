const express = require('express');
const multer = require('multer');
const pdfparse = require('pdf-parse');
const { QdrantClient } = require('@qdrant/js-client-rest');
const fs = require('fs');
const { ChatOllama, OllamaEmbeddings } = require('@langchain/ollama');

const app = express();
app.use(express.json());
const port = 3000;  

const model = new ChatOllama({
    model: "llama3.2",
    temperature: 0
});

const embeddings = new OllamaEmbeddings({
    model: "nomic-embed-text",
});

const qdrant = new QdrantClient({
    url: process.env.QDRANT_URL || 'https://87611927-b01f-4ea9-8e9c-46e6330ccfb2.eu-west-1-0.aws.cloud.qdrant.io:6333',
    apiKey: process.env.QDRANT_API_KEY || 'YOUR_ACTUAL_API_KEY_HERE',
    checkCompatibility: false 
});

async function createEmbeddings(text) {
    try {
        const embedding = await embeddings.embedQuery(text);    
        return embedding;
        console.log("Embedding created successfully for text:", text.substring(0, 50) + "...");
    }
    catch (error) {
        console.error("Error creating embedding:", error);
    }
}   

function cosineSimilarity(vecA, vecB){
    let dotProduct = 0;
    for(let i=0; i< vecA.length; i++){
        dotProduct += vecA[i] * vecB[i];
    }
    return dotProduct;
}

const upload = multer({ dest: 'uploads/' });

app.get('/create-collection', async (req, res) => {
    try {
        await qdrant.createCollection('pdf_docs', {
            vectors: {
                size: 768,
                distance: "Cosine",
            },
        });
        res.send('collection is created');
    } 
   
         catch (error) {
        // Change this line to print the FULL error object
        console.error("FULL ERROR DETAILS:", error); 
        res.status(500).json({ error: "Failed to process request." });
    }
    
});

app.post('/upload', upload.single('pdf'), async (req, res) => {
    try {
        console.log("1. File uploaded:", req.file.originalname);

        const pdfBuffer = fs.readFileSync(req.file.path);
        const pdfData = await pdfparse(pdfBuffer);
        const text = pdfData.text;
       
        const chunks = text.split('\n\n')
            .map(chunk => chunk.trim())
            .filter(chunk => chunk.length > 50);

        if (chunks.length === 0) {
            return res.status(400).json({ error: "Could not extract readable text from this PDF." });
        }

        const chunkembeddings = [];
        for (const chunk of chunks) {
            const emb = await createEmbeddings(chunk);
            chunkembeddings.push({
                text: chunk,
                embedding: emb
            });
        }

        // --- QDRANT UPLOAD FIXED ---
        // We wrap the upsert in an async function to prevent top-level await errors
        async function saveToQdrant() {
            console.log("4. Uploading chunks to Qdrant...");
            const points = chunkembeddings.map((item, index) => ({
                id: index + 1,
                vector: item.embedding,
                payload: { text: item.text },
            }));

            await qdrant.upsert("pdf_docs", {
                wait: true,
                points: points,
            });
            console.log("5. Successfully saved to Qdrant!");
        }

        await saveToQdrant();
        // ----------------------------

        const question = req.body.question || "What is this document about?";
        const questionEmbedding = await createEmbeddings(question);
        
        let bestchunk = null;
        let bestscore = -Infinity;
        
        for (const chunk of chunkembeddings) {
            const score = cosineSimilarity(questionEmbedding, chunk.embedding);
            if (score > bestscore) {
                bestscore = score;
                bestchunk = chunk.text;
            }
        }
        
        console.log("Best matching chunk from manual search:", bestchunk.substring(0, 50) + "...");
        
        const response = await model.invoke(`Answer the following question based on the provided text:\n\nText: ${bestchunk}\n\nQuestion: ${question}`); 
        
        res.json({ 
            message: 'File uploaded, saved to Qdrant, and processed by Ollama!', 
            totalChunks: chunks.length,
            aiResponse: response.content
        });

    } catch (error) {
        console.error("Error:", error.message);
        res.status(500).json({ error: "Failed to read the PDF, save to Qdrant, or call Ollama." });
    }
});

app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});