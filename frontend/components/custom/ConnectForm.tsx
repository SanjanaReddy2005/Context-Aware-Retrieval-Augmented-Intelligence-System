"use client";
import { Button } from "@/components/ui/button";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Bot, Loader2 } from "lucide-react";
import axios from "axios";
import { useState } from "react";

const dbSchema = z.object({
  user: z.string().min(1).max(50),
  password: z.string().min(1).max(50),
  database: z.string().min(1).max(50),
  host: z.string().min(1).max(50),
  port: z.string().min(1).max(65535),
});

interface ConnectFormProps {
  setConnection: (data: string) => void;
}

const ConnectForm = ({ setConnection }: ConnectFormProps) => {
  const [loading, setLoading] = useState(false);
  const form = useForm<z.infer<typeof dbSchema>>({
    resolver: zodResolver(dbSchema),
    defaultValues: {
      user: "",
      password: "",
      database: "",
      host: "",
      port: "",
    },
  });

  async function onSubmit(values: z.infer<typeof dbSchema>) {
    setLoading(true);
    try {
      console.log(values);
      const response = await axios.post(
        "http://localhost:8000/api/connect-to-db",
        values
      );
      console.log(response.data.message);
      setConnection(response.data.message);
    } catch (error) {
      console.error(error);
    }
    setLoading(false);
  }

  return (
    <div>
      <Form {...form}>
        <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-8">
          <div className="grid lg:grid-cols-3 md:grid-cols-2 w-full gap-8">
            <FormField
              control={form.control}
              name="user"
              render={({ field }) => (
                <FormItem>
                  <FormLabel className="text-gray-700 dark:text-white">
                    User
                  </FormLabel>
                  <FormControl>
                    <Input placeholder="Enter User" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="password"
              render={({ field }) => (
                <FormItem>
                  <FormLabel className="text-gray-700 dark:text-white">
                    Password
                  </FormLabel>
                  <FormControl>
                    <Input placeholder="Enter Password" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="host"
              render={({ field }) => (
                <FormItem>
                  <FormLabel className="text-gray-700 dark:text-white">
                    Host
                  </FormLabel>
                  <FormControl>
                    <Input placeholder="Enter Host" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="port"
              render={({ field }) => (
                <FormItem>
                  <FormLabel className="text-gray-700 dark:text-white">
                    Port
                  </FormLabel>
                  <FormControl>
                    <Input placeholder="Enter Port" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="database"
              render={({ field }) => (
                <FormItem>
                  <FormLabel className="text-gray-700 dark:text-white">
                    Database
                  </FormLabel>
                  <FormControl>
                    <Input placeholder="Enter Database" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
          </div>
          <Button
          disabled={loading}
            type="submit"
            className="text-lg px-3 py-4 gap-2 bg-gradient-to-br dark:text-white from-amber-300 to to-orange-600 hover:scale-105 transition-all duration-300"
          >
            Connect
            {!loading && <Bot size={24} className="hover:animate-spin" />}
            {loading && <Loader2 size={24} className="animate-spin" />}
          </Button>
        </form>
      </Form>
    </div>
  );
};

export default ConnectForm;
