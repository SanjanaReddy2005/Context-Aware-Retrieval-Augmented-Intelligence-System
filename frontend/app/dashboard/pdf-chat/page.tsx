"use client";
import React, { useState } from "react";
import { Bot } from "lucide-react";
import ChatContainer from "@/components/custom/ChatContainer";
import UploadForm from "@/components/custom/UploadForm";

function PDFChat() {
  const [connection, setConnection] = useState<string>("");
  const url = "http://localhost:8000/api/get-pdf-response";
  
  return (
    <div className="flex flex-col items-center dark:bg-darkPrimary bg-white gap-2 z-0">
      <div className="relative pt-12">
        <div className="relative flex flex-col items-center max-w-screen-xl mx-auto px-8 text-center">
          <h1 className="bg-gradient-to-br from-amber-300 to to-orange-600 text-transparent bg-clip-text text-3xl md:text-5xl font-bold leading-tight mb-6">
            Chat with PDF Files
          </h1>
          <p className="text-gray-500 dark:text-white text-lg max-w-2xl text-center mb-8">
            Effortlessly chat with your PDF files—extract, explore, and
            understand information in real-time.
          </p>
        </div>
      </div>
      <div className="w-full sm:px-16 px-8 py-4 mb-2">
        <UploadForm setConnection={setConnection} />
      </div>
      {connection && (
        <div className="bg-green-300 px-4 py-1 mx-4 border-2 border-green-700 rounded-md mb-4">
          <p className="text-gray-900 text-sm">{connection}</p>
        </div>
      )}
      <div className="flex flex-col gap-4 w-full sm:px-16 px-8 rounded-full mb-4">
        <div className="flex gap-2 items-center text-gray-700 dark:text-white">
          <h1 className="text-2xl font-bold ">Start Chatting</h1>
          <Bot size={28} />
        </div>
        <ChatContainer connection={connection} url={url} placeholder="Ask a question to your PDF" />
      </div>
    </div>
  );
}

export default PDFChat;
