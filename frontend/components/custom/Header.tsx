import ThemeToggle from "@/components/custom/ThemeToggle";
import { Menu, Search } from "lucide-react";
import React from "react";
import { UserButton } from "@clerk/nextjs";

interface PROPS {
  setShowSidebar: (value: boolean) => void;
}

function Header({ setShowSidebar }: PROPS) {
  return (
    <div className="p-5 shadow-sm border-b-2 flex flex-row gap-3 dark:bg-darkSecondary bg-white justify-between items-center">
      <div className="lg:hidden">
        <Menu
          className="text-darkPrimary dark:text-white"
          size={32}
          onClick={() => {
            setShowSidebar(true);
          }}
        />
      </div>
      <div className="sm:flex sm:flex-row hidden dark:bg-gray-700 gap-2 items-center p-2 border rounded-full max-w-lg">
        <Search size={24} className="text-gray-700 dark:text-white" />
        <input
          type="text"
          placeholder="Search..."
          className="dark:text-white text-gray-700 dark:bg-gray-700 border-none outline-none"
        />
      </div>
      <div className="flex gap-5 items-center">
        <ThemeToggle />
        <UserButton />
      </div>
    </div>
  );
}

export default Header;
