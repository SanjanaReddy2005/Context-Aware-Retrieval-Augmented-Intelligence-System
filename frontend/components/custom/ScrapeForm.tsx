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

const urlSchema = z.object({
  url: z.string().min(1).max(200),
});

interface ConnectFormProps {
  setConnection: (data: string) => void;
}

const ScrapeForm = ({ setConnection }: ConnectFormProps) => {
  const [loading, setLoading] = useState(false);
  const form = useForm<z.infer<typeof urlSchema>>({
    resolver: zodResolver(urlSchema),
    defaultValues: {
      url: ""
    },
  });

  async function onSubmit(values: z.infer<typeof urlSchema>) {
    setLoading(true);
    try {
      console.log(values);
      const response = await axios.post(
        "http://localhost:8000/api/scrape-website",
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
          <div>
            <FormField
              control={form.control}
              name="url"
              render={({ field }) => (
                <FormItem>
                  <FormLabel className="text-gray-700 dark:text-white">
                    Website Url
                  </FormLabel>
                  <FormControl>
                    <Input placeholder="Enter Website Url" {...field} />
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
            Scrape
            {!loading && <Bot size={24} className="hover:animate-spin" />}
            {loading && <Loader2 size={24} className="animate-spin" />}
          </Button>
        </form>
      </Form>
    </div>
  );
};

export default ScrapeForm;
