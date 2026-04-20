"use client";
import React from "react";
import { useUser } from "@clerk/nextjs";
import { Bot, Database, FileText } from "lucide-react";
import { Button } from "@/components/ui/button";
import Link from "next/link";

function Dashboard() {
  const { user } = useUser();
  return (
    <div className="flex flex-col items-center mt-12 dark:bg-darkPrimary bg-white gap-2 z-0 pb-32">
      <div className="relative pt-12">
        <div className="relative flex flex-col items-center max-w-screen-xl mx-auto px-8 text-center">
          <h1 className="bg-gradient-to-br from-amber-300 to to-orange-600 text-transparent bg-clip-text text-3xl md:text-5xl font-bold leading-tight mb-6">
            Welcome {user?.firstName || "User"} !
          </h1>
          <p className="text-gray-500 dark:text-white text-lg max-w-2xl text-center mb-8">
            Unlock deeper insights from your <span className="text-green-300">PDFs</span> and <span className="text-green-300">SQL Databases</span> through
            intelligent, real-time conversations with <span className="text-green-300">Doom AI</span>—transforming data
            into knowledge effortlessly.
          </p>
        </div>
      </div>
      <div className="grid lg:grid-cols-2 md:grid-cols-2 max-w-5xl mx-auto gap-6 px-4 my-6">
        <div className="hover:bg-gray-100 dark:hover:bg-darkSecondary p-6 rounded-md cursor-pointer border-2">
          <div className="flex gap-4 items-center">
            <FileText
              size={48}
              className="mb-3 inline-block bg-gray-700 text-white p-2 rounded-md"
            />
            <h3 className="text-xl font-bold mb-2 dark:text-white text-gray-800">
              PDF Files
            </h3>
          </div>
          <p className="text-md dark:text-gray-300 mb-4 text-gray-600">
            Effortlessly chat with your PDF files—extract, explore, and
            understand information in real-time.
          </p>
          <Link href="/dashboard/pdf-chat">
          <Button
            type="button"
            className="text-lg px-3 py-4 gap-2 bg-gradient-to-br dark:text-white from-amber-300 to to-orange-600 hover:scale-105 transition-all duration-300"
          >
            Start
            <Bot size={24} className="hover:animate-spin" />
          </Button>
          </Link>
        </div>
        <div className="hover:bg-gray-100 dark:hover:bg-darkSecondary p-6 rounded-md cursor-pointer border-2">
        <div className="flex gap-4 items-center">
          <Database
            size={48}
            className="mb-3 inline-block bg-gray-700 text-white p-2 rounded-md"
          />
          <h3 className="text-xl font-bold mb-2 dark:text-white text-gray-800">
            MySQL Database
          </h3>
        </div>
          <p className="text-md dark:text-gray-300 mb-4 text-gray-600">
            Seamlessly chat with your SQL databases—query, analyze, and
            visualize data in real-time.
          </p>
          <Link href="/dashboard/sql-chat">
          <Button
            type="button"
            className="text-lg px-3 py-4 gap-2 bg-gradient-to-br dark:text-white from-amber-300 to to-orange-600 hover:scale-105 transition-all duration-300"
          >
            Query
            <Bot size={24} className="hover:animate-spin" />
          </Button>
          </Link>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
