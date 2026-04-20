"use client";

import { Database, FileText, Home, PanelsTopLeftIcon, Settings, X } from "lucide-react";
import { usePathname } from "next/navigation";
import React from "react";
import Link from "next/link";

interface MENU {
  title: string;
  icon: any;
  path: string;
}

interface PROPS {
  showSidebar: boolean;
  setShowSidebar: (value: any) => void;
}

function SideNav({ showSidebar, setShowSidebar }: PROPS) {
  const path = usePathname();

  const MenuList: MENU[] = [
    {
      title: "Home",
      icon: Home,
      path: "/dashboard",
    },
    {
      title: "PDF Chat",
      icon: FileText,
      path: "/dashboard/pdf-chat",
    },
    {
      title: "MySQL Chat",
      icon: Database,
      path: "/dashboard/sql-chat",
    },
    {
      title: "Website Chat",
      icon: PanelsTopLeftIcon,
      path: "/dashboard/website-chat"
    },
    {
      title: "Settings",
      icon: Settings,
      path: "/dashboard/settings",
    },
  ];

  return (
    <div
      className={`${
        showSidebar ? "ml-0" : "ml-[-340px]"
      } lg:ml-0 w-72 lg:w-64 transition-[margin-left] ease-in-out duration-500 h-screen fixed dark:bg-darkSecondary p-5 shadow-sm border bg-white z-10`}
    >
      <div className="flex flex-row justify-between items-center">
        <div className="justify-center flex items-center gap-1">
          <span className="font-bold text-xl text-gray-700 dark:text-white">
            🤖 Doom AI
          </span>
        </div>
        <X
          size={24}
          className="lg:hidden cursor-pointer hover:text-gray-700"
          onClick={() => setShowSidebar(false)}
        />
      </div>
      <hr className="my-5 border" />
      <div className="mt-3">
        {MenuList.map((menu, index) => (
          <Link
            key={index}
            href={menu.path}
            onClick={() => setShowSidebar(false)}
          >
            <div
              className={`flex items-center mb-3 p-3 gap-2 cursor-pointer text-gray-700 dark:text-white hover:bg-gray-700 hover:text-white rounded-lg ${
                path === menu.path && "bg-gray-700 text-white"
              }`}
            >
              <menu.icon size={24} />
              <span className="text-lg">{menu.title}</span>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}

export default SideNav;
