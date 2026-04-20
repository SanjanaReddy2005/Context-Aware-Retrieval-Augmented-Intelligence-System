import ThemeToggle from "@/components/custom/ThemeToggle";
import { Button } from "@/components/ui/button";
import {
  ArrowRight,
  Database,
  FileText
} from "lucide-react";
import Image from "next/image";
import Link from "next/link";

export default function Home() {

  return (
    <>
      <header className="border-b py-4 px-4 sm:px-10 dark:bg-darkSecondary bg-white tracking-wide relative z-50">
        <div className="max-w-7xl w-full mx-auto flex flex-wrap items-center gap-4">
          <Link
            href="/"
            className="flex gap-1 items-center justify-center font-bold text-lg"
          >
            <span className="text-gray-700 dark:text-white text-xl">🤖 Doom AI</span>
          </Link>
          <div className="flex gap-8 items-center ml-auto">
            <ThemeToggle />
            <Link href="/dashboard">
              <Button
                variant="outline"
                className="flex items-center gap-2 transition-all text-md font-semibold rounded-md px-2 py-5"
              >
                SignIn
                <ArrowRight size={24} />
              </Button>
            </Link>
          </div>
        </div>
      </header>

      <div className="flex flex-col items-center dark:bg-darkPrimary bg-white">
          <div className="relative pt-28 pb-12">
            <div className="absolute inset-0">
              <Image
                src="https://preline.co/assets/svg/examples/polygon-bg-element.svg"
                alt="Background Image"
                width={1920}
                height={1080}
                className="w-full h-full object-cover dark:hidden"
              />
              <Image
                src="https://preline.co/assets/svg/examples-dark/polygon-bg-element.svg"
                alt="Background Image"
                width={1920}
                height={1080}
                className="w-full h-full object-cover hidden dark:block"
              />
            </div>

            <div className="relative flex flex-col items-center max-w-screen-xl mx-auto px-8 z-10 text-center">
              <h1 className="text-gray-700 dark:text-white text-4xl md:text-6xl font-bold leading-tight mb-6">
                Doom AI:{" "}
                <span className="bg-gradient-to-br from-amber-300 to to-orange-600 text-transparent bg-clip-text">
                  Intelligent Chat
                </span>
              </h1>
              <p className="text-gray-500 dark:text-white text-lg max-w-2xl text-center md:text-xl mb-12">
                Unlock deeper insights from your PDFs and SQL databases through
                intelligent, real-time conversations with Doom AI—transforming
                data into knowledge effortlessly.
              </p>
              <Link href="/dashboard">
                <Button
                  type="button"
                  className="text-lg px-4 py-6 gap-2 bg-gradient-to-br dark:text-white from-amber-300 to to-orange-600 hover:scale-105 transition-all duration-300"
                >
                  Get Started
                  <ArrowRight size={24} className="hover:animate-spin" />
                </Button>
              </Link>
            </div>
          </div>

          <div className="grid lg:grid-cols-2 md:grid-cols-2 max-w-5xl mx-auto gap-6 px-4 my-12">
            <div className="hover:bg-gray-100 dark:hover:bg-gray-700 p-6 rounded-md hover:scale-105 transition-all duration-300 cursor-pointer">
              <FileText
                size={48}
                className="mb-3 inline-block bg-gray-700 text-white p-2 rounded-md"
              />
              <h3 className="text-xl font-bold mb-2 dark:text-white text-gray-800">
                PDF Files
              </h3>
              <p className="text-md dark:text-gray-300 text-gray-600">
                Effortlessly chat with your PDF files—extract, explore, and
                understand information in real-time.
              </p>
              <a
                href="/"
                className="text-orange-500 font-bold inline-block text-sm mt-4"
              >
                Learn more
              </a>
            </div>
            <div className="hover:bg-gray-100 dark:hover:bg-gray-700 p-6 rounded-md hover:scale-105 transition-all duration-300 cursor-pointer">
              <Database
                size={48}
                className="mb-3 inline-block bg-gray-700 text-white p-2 rounded-md"
              />
              <h3 className="text-xl font-bold mb-2 dark:text-white text-gray-800">
                MySQL Database
              </h3>
              <p className="text-md dark:text-gray-300 text-gray-600">
                Seamlessly chat with your SQL databases—query, analyze, and
                visualize data in real-time.
              </p>
              <a
                href="/"
                className="text-orange-500 font-bold inline-block text-sm mt-4"
              >
                Learn more
              </a>
            </div>
          </div>
        </div>
    </>
  );
}
