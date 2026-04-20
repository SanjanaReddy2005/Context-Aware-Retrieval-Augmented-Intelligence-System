"use client";
import React, { useState } from "react";
import Header from "@/components/custom/Header";
import Sidebar from "@/components/custom/Sidebar";

function DashboardLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const [showSideBar, setShowSideBar] = useState<boolean>(false);

  return (
          <div className="h-full dark:bg-darkPrimary bg-slate-100">
            <div>
              <Sidebar
                showSidebar={showSideBar}
                setShowSidebar={setShowSideBar}
              />
            </div>
            <div className="lg:ml-64">
              <Header setShowSidebar={setShowSideBar} />
              {children}
            </div>
          </div>
  );
}

export default DashboardLayout;
